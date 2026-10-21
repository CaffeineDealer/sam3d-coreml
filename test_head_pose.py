import sys, types, os
import numpy as np
import torch

class _Stub(types.ModuleType):  # fake pyrender, same as before
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

sys.modules["pyrender"] = _Stub("pyrender")

import coremltools as ct
import coremltools.converters.mil.frontend.torch.ops as _tops  # coremltools' op converters
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.mil import Var

mb = _tops.mb

# --- patch 1: int/float/bool casts of 1-element arrays (newer numpy rejects them) ---
def _cast_fixed(context, node, dtype, dtype_name):
    x = _tops._get_inputs(context, node, expected=1)[0]
    if x.can_be_folded_to_const():
        v = x.val
        v = v.item() if hasattr(v, "item") else v
        res = mb.const(val=dtype(v), name=node.name)
    elif len(x.shape) > 0:
        x = mb.squeeze(x=x, name=node.name + "_item")
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    else:
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)

_tops._cast = _cast_fixed

# --- patch 2: skip the optional cleanup pass that crashes on the rig ---
torch._C._jit_pass_canonicalize_graph_fuser_ops = lambda graph: None

# --- patch 3: linalg_cross reuses coremltools' own cross product ---
_TORCH_OPS_REGISTRY.set_func_by_name(_tops.cross, "linalg_cross")

# --- patch 4: general torch.diagonal (any offset, any two dims) ---
def _int(a):  # a constant index argument, as a Python int
    return int(a.val) if isinstance(a, Var) else int(a)

@register_torch_op(torch_alias=["diagonal_copy"], override=True)
def diagonal(context, node):
    inputs = _tops._get_inputs(context, node, min_expected=1)
    x = inputs[0]
    offset = _int(inputs[1]) if len(inputs) > 1 else 0
    dim1 = _int(inputs[2]) if len(inputs) > 2 else 0
    dim2 = _int(inputs[3]) if len(inputs) > 3 else 1
    rank = x.rank
    d1, d2 = dim1 % rank, dim2 % rank
    rest = [i for i in range(rank) if i not in (d1, d2)]
    xt = mb.transpose(x=x, perm=rest + [d1, d2])  # the two dims go last: [..., n1, n2]
    n1, n2 = int(x.shape[d1]), int(x.shape[d2])
    mask = np.zeros((n1, n2), dtype=np.float32)
    for i in range(n1):
        if 0 <= i + offset < n2:
            mask[i, i + offset] = 1.0
    d = mb.reduce_sum(x=mb.mul(x=xt, y=mask), axes=[-1], keep_dims=False)  # [..., n1]
    start, stop = max(0, -offset), min(n1, n2 - offset)
    nb = rank - 2
    res = mb.slice_by_index(x=d, begin=[0] * nb + [start], end=[0] * nb + [stop],
                            begin_mask=[True] * nb + [False], end_mask=[True] * nb + [False], name=node.name)
    context.add(res)

# --- patch 5: view() to a scalar, without the size assert that fails on unknown-size inputs ---
_orig_view = _tops.view

@register_torch_op(torch_alias=["view_copy", "_unsafe_view", "reshape"], override=True)
def view(context, node):
    inputs = _tops._get_inputs(context, node, expected=2)
    x, shape = inputs[0], inputs[1]
    # patch 21: the scalar shape can also arrive as an empty Python list, not only an empty Var
    if (isinstance(shape, Var) and int(np.prod(shape.shape)) == 0) or (isinstance(shape, list) and len(shape) == 0):  # reshape to a scalar
        print("VIEW-SCALAR input shape:", x.shape)
        flat = mb.reshape(x=x, shape=[1])  # assumes it holds exactly one element
        context.add(mb.squeeze(x=flat, axes=[0], name=node.name))
        return
    return _orig_view(context, node)

# --- patch 6: hypot, index_add, index_copy ---
@register_torch_op
def hypot(context, node):
    a, b = _tops._get_inputs(context, node, expected=2)
    sq = mb.add(x=mb.mul(x=a, y=a), y=mb.mul(x=b, y=b))
    context.add(mb.sqrt(x=sq, name=node.name))

def _index_scatter(context, node, mode):  # index_add / index_copy with a 1-D index along dim
    inputs = _tops._get_inputs(context, node, min_expected=4)
    x = inputs[0]
    dim = _int(inputs[1]) % x.rank
    index = mb.cast(x=inputs[2], dtype="int32")
    src = inputs[3]
    if mode == "add" and len(inputs) > 4:  # optional alpha multiplier
        alpha = float(inputs[4].val if isinstance(inputs[4], Var) else inputs[4])
        if alpha != 1.0:
            src = mb.mul(x=src, y=alpha)
    context.add(mb.scatter(data=x, indices=index, updates=src, axis=dim, mode=mode, name=node.name))

@register_torch_op
def index_add(context, node):
    _index_scatter(context, node, "add")

# --- patch 28: index_copy_ inside a loop writes into a tensor defined outside it (the rig's forward kinematics); patch 27 can't
#     reroute inside blocks, so bind the result back to the target's name in the scope that defined it ---
def _rebind_outer(context, name, var):
    scopes = context._current_graph  # list of {torch name: Var}, innermost last
    if name in scopes[-1]:  # defined in this same block: an ordinary write, not handled here
        return False
    for scope in reversed(scopes[:-1]):
        if name in scope:
            scope[name] = var  # later reads (next unrolled pass, code after the loop) now get the written tensor
            return True
    return False

@register_torch_op
def index_copy(context, node):  # also serves index_copy_, coremltools treats in-place ops as functional
    _index_scatter(context, node, "update")
    if _rebind_outer(context, node.inputs[0], context[node.name]):
        print("INDEX-COPY-REBIND", node.inputs[0])


# --- patch 8: x[..., idx_list] = values (and constant bool masks) via scatter along the axis ---
_orig_index_put = _tops.index_put

@register_torch_op(override=True)
def index_put(context, node):
    inputs = _tops._get_inputs(context, node, expected=(3, 4))
    x, indices, values = inputs[0], inputs[1], inputs[2]
    accumulate = inputs[3] if len(inputs) > 3 else False
    accumulate = accumulate.val if isinstance(accumulate, Var) else accumulate
    real = [(d, i) for d, i in enumerate(indices) if i is not None] if isinstance(indices, list) else []
    if len(real) == 1 and len(real[0][1].shape) == 1 and len(real) < len(indices):  # exactly one index vector, with whole slices before it
        d, idx = real[0]
        prim = idx.sym_type.get_primitive()
        if _tops.types.is_bool(prim) and idx.val is not None:  # constant bool mask -> positions of the True entries
            pos = np.nonzero(idx.val)[0].astype(np.int32)
            if len(pos) == 0:  # all False: nothing to assign
                context.add(mb.identity(x=x, name=node.name))
                return
            idx = mb.const(val=pos)
        elif _tops.types.is_int(prim):
            idx = mb.cast(x=idx, dtype="int32")
        else:
            return _orig_index_put(context, node)
        print("INDEX-PUT-VECTOR dim", d, "x", x.shape, "index", idx.shape, "values", values.shape)
        want = tuple(x.shape[:d]) + (idx.shape[0],) + tuple(x.shape[d + 1:])  # the shape torch would assign into
        if tuple(values.shape) != want:
            values = _tops._utils.pymil_broadcast_to(values, want, values.name + "_broadcasted")
        x, values = _tops.promote_input_dtypes([x, values])
        context.add(mb.scatter(data=x, indices=idx, updates=values, axis=d,
                               mode="add" if accumulate else "update", name=node.name))
        return
    return _orig_index_put(context, node)

# --- patch 9: len() of a list or a tensor ---
def _len_op(context, node):
    x = _tops._get_inputs(context, node, expected=1)[0]
    if isinstance(x, (list, tuple)):  # Python list: its length is known now
        res = mb.const(val=np.int32(len(x)), name=node.name)
    elif isinstance(x, _tops.ListVar):  # MIL list
        res = mb.list_length(ls=x, name=node.name)
    else:  # tensor: size of its first dimension
        res = _tops._list_select(mb.shape(x=x), 0)
    context.add(res, node.name)

_TORCH_OPS_REGISTRY.set_func_by_name(_len_op, "len")  # registered under the name "len" without shadowing Python's len

  # --- patch 10: the two einsum equations in the rig, as matrix multiplies ---
_orig_einsum = _tops.einsum

@register_torch_op(override=True)
def einsum(context, node):
    if context.frontend == _tops.TorchFrontend.TORCHSCRIPT:
        eq = str(context[node.inputs[0]].val).replace(" ", "")  # ignore spaces in the equation
        if eq in ("nvd,...n->...vd", "dn,...n->...d"):
            a, b = _tops.promote_input_dtypes(context[node.inputs[1]])  # a: the fixed matrix, b: the batch of vectors [..., n]
            if not (_tops.any_symbolic(a.shape) or _tops.any_symbolic(b.shape)):
                lead = [int(s) for s in b.shape[:-1]]  # leading (batch) dims of b
                n = int(b.shape[-1])
                b2 = mb.reshape(x=b, shape=[-1, n])
                if eq == "nvd,...n->...vd":  # blend shapes: a is [n, v, d]
                    v, d = int(a.shape[1]), int(a.shape[2])
                    prod = mb.matmul(x=b2, y=mb.reshape(x=a, shape=[n, v * d]))
                    out = mb.reshape(x=prod, shape=lead + [v, d], name=node.name)
                else:  # parameter transform: a is [d, n]
                    d = int(a.shape[0])
                    prod = mb.matmul(x=b2, y=a, transpose_y=True)
                    out = mb.reshape(x=prod, shape=lead + [d], name=node.name)
                context.add(out)
                return
    return _orig_einsum(context, node)  # every other equation goes through coremltools' own converter

# --- patch 11: pick an item out of a constant list that coremltools turned into a tensor ---
_orig_getitem = _tops.getitem

@register_torch_op(override=True)
def getitem(context, node):
    x, i = _tops._get_inputs(context, node, expected=2)
    if isinstance(x, Var) and not isinstance(x, _tops.ListVar) and len(x.shape) >= 1 and i.val is not None:
        idx = int(i.val)
        if x.can_be_folded_to_const() and len(x.shape) == 1:  # a constant list became a constant vector: take the element
            print("GETITEM-CONST-VECTOR index", idx, "of", x.shape)
            context.add(mb.const(val=x.val[idx], name=node.name))
            return
        if idx != 0:  # any other tensor with a non-zero index, which the original converter rejects
            print("GETITEM-TENSOR index", idx, "of", x.shape)
            context.add(mb.gather(x=x, indices=idx, axis=0, name=node.name))
            return
    return _orig_getitem(context, node)  # lists, tuples and index 0 behave as before

# --- patch 12: slicing a constant list, e.g. parents[:k] (4 inputs: list, start, end, step) ---
_orig_slice = _tops.slice

def _slice_op(context, node):
    inputs = _tops._get_inputs(context, node, expected=(1, 2, 3, 4, 5))
    if len(inputs) == 4:  # a tensor slice always has 5 inputs, so 4 means a list slice
        x, start, end, step = inputs
        pick = lambda v: None if v is None else int(v.val if isinstance(v, Var) else v)
        start, end, step = pick(start), pick(end), pick(step) or 1
        if isinstance(x, (list, tuple)):  # Python list: slice it now
            print("SLICE-LIST", start, end, step, "of a python list")
            context.add(type(x)(x[start:end:step]), node.name)
            return
        if isinstance(x, Var) and x.can_be_folded_to_const():  # constant vector: slice its values now
            print("SLICE-LIST", start, end, step, "of", x.shape)
            context.add(mb.const(val=x.val[start:end:step], name=node.name))
            return
    return _orig_slice(context, node)  # real tensor slices behave as before

_TORCH_OPS_REGISTRY.set_func_by_name(_slice_op, "slice")  # registered as "slice" without shadowing Python's slice

# --- patch 13: aten::list copies a list, so pass it through ---
def _list_op(context, node):
    x = _tops._get_inputs(context, node, expected=1)[0]
    context.add(list(x) if isinstance(x, (list, tuple)) else x, node.name)  # Python list: copy it; tensor or MIL list: unchanged

_TORCH_OPS_REGISTRY.set_func_by_name(_list_op, "list")  # registered as "list" without shadowing Python's list

# --- patch 14: list + list must be joined, but coremltools adds them element by element ([1] + [-1, 7] became [0, 8]) ---
_LIST_IDS = set()  # ids of constant vectors that stand for Python lists of ints, such as shapes

def _mark(fn, only_if=lambda node: True):  # run the converter, then remember its output as a list
    def wrapped(context, node):
        fn(context, node)
        if only_if(node):
            _LIST_IDS.add(id(context[node.name]))
    return wrapped

_TORCH_OPS_REGISTRY.set_func_by_name(_mark(_slice_op, lambda n: len(n.inputs) == 4), "slice")  # list slices only
_TORCH_OPS_REGISTRY.set_func_by_name(_mark(_list_op), "list")
_TORCH_OPS_REGISTRY.set_func_by_name(_mark(_tops.size, lambda n: len(n.inputs) == 1), "size")  # the full shape of a tensor

_orig_add = _tops.add

@register_torch_op(override=True)
def add(context, node):
    ins = _tops._get_inputs(context, node)
    if (len(ins) >= 2
            and all(isinstance(v, Var) and v.can_be_folded_to_const() and len(v.shape) == 1 for v in ins[:2])
            and (id(ins[0]) in _LIST_IDS or id(ins[1]) in _LIST_IDS)):  # two constant vectors, at least one known to be a list
        print("LIST-CONCAT", ins[0].val, ins[1].val)
        res = mb.const(val=np.concatenate([ins[0].val, ins[1].val]), name=node.name)
        _LIST_IDS.add(id(res))  # the joined list is a list too
        context.add(res)
        return
    return _orig_add(context, node)  # tensor additions behave as before

# --- patch 30: coremltools' convert_block stops a block as soon as its outputs exist (early_exit=True), so in-place writes in a
#     loop body or if branch (the rig's index_copy_ in forward kinematics) are never converted. Same steps, every op converted ---
def _convert_block_all(context, block, inputs):
    context.push((block.inputs, inputs))  # new name scope holding the block's inputs, as in convert_block
    _tops.convert_nodes(context, block, early_exit=False)  # convert every op, including writes nothing in the block reads
    outputs = [context[o] for o in block.outputs]
    context.pop()
    return outputs

# --- patch 15: an if with a condition known at conversion time converts only the branch that is taken ---
_orig_if = _tops._if

def _if_op(context, node):
    cond = _tops._get_inputs(context, node, expected=1)[0]
    if isinstance(cond, Var) and cond.val is not None:  # known now: PyTorch would run only one branch
        taken = bool(cond.val)
        print("STATIC-IF", "true" if taken else "false", "branch")
        outs = _convert_block_all(context, node.blocks[0] if taken else node.blocks[1], [])  # patch 30: convert every op
        for out_name, out_var in zip(node.outputs, outs):
            context.add(out_var, torch_name=out_name)
        return
    return _orig_if(context, node)  # conditions only known at run time keep both branches

_TORCH_OPS_REGISTRY.set_func_by_name(_if_op, "if")
_TORCH_OPS_REGISTRY.set_func_by_name(_if_op, "_if")  # the converter is registered under both names

# --- patch 16: a loop whose condition is known at conversion time is unrolled instead of becoming a while_loop ---
_orig_loop = _tops.loop

def _loop_op(context, node):
    inputs = _tops._get_inputs(context, node)
    max_iter, cond, carried = inputs[0], inputs[1], list(inputs[2:])
    cond_val = cond.val if isinstance(cond, Var) else cond
    max_val = max_iter.val if isinstance(max_iter, Var) else max_iter
    if cond_val is None or (max_iter is not None and max_val is None):  # decided at run time: keep the original
        return _orig_loop(context, node)
    i = 0
    while bool(cond_val) and (max_val is None or i < int(max_val)):
        body_in = (mb.const(val=i),) + tuple(carried)  # the loop counter, then the carried values
        res = _convert_block_all(context, node.blocks[0], body_in)  # patch 30: convert every op, so index_copy_ runs
        new_cond, carried = res[0], list(res[1:])  # the body returns: next condition, then the new values
        cond_val = new_cond.val if isinstance(new_cond, Var) else new_cond
        if cond_val is None or i > 64:
            raise ValueError("loop can't be unrolled: condition unknown or too many passes")
        i += 1
    print("UNROLLED-LOOP", i, "passes")
    for out_name, out_var in zip(node.outputs, carried):
        context.add(out_var, torch_name=out_name)

_TORCH_OPS_REGISTRY.set_func_by_name(_loop_op, "loop")

# --- patch 17: split where the sizes arrive as a str or list instead of a tensor ---
import ast
_orig_split = _tops.split

def _split_op(context, node):
    ins = _tops._get_inputs(context, node, min_expected=2)
    sizes = ins[1]
    if isinstance(ins[0], Var) and isinstance(sizes, (str, list, tuple)):
        print("SPLIT-SIZES", type(sizes).__name__, repr(sizes)[:120])
        if isinstance(sizes, str):
            try:
                sizes = ast.literal_eval(sizes)  # the list written out as text
            except Exception:  # not readable: take the real list from the loaded model
                sizes = list(model.head_pose.mhr.character_torch.skeleton._pmi_buffer_sizes)
        sizes = [int(s.val) if isinstance(s, Var) else int(s) for s in sizes]
        dim = ins[2].val if len(ins) > 2 and isinstance(ins[2], Var) else (ins[2] if len(ins) > 2 else 0)
        res = mb.split(x=ins[0], split_sizes=np.array(sizes, dtype=np.int32), axis=int(dim), name=node.name)
        context.add(res, torch_name=node.name)
        return
    return _orig_split(context, node)

_TORCH_OPS_REGISTRY.set_func_by_name(_split_op, "split")
_TORCH_OPS_REGISTRY.set_func_by_name(_split_op, "split_with_sizes")

# --- patch 18: sparse_coo_tensor with stored indices and weights becomes a dense constant matrix ---
def _sparse_coo_op(context, node):
    ins = _tops._get_inputs(context, node, min_expected=3)
    idx, vals, size = ins[0], ins[1], ins[2]
    if not (isinstance(idx, Var) and idx.val is not None and isinstance(vals, Var) and vals.val is not None):
        raise NotImplementedError("sparse_coo_tensor with indices or values only known at run time")
    dims = size.val.tolist() if isinstance(size, Var) else list(size)
    dims = [int(d.val) if isinstance(d, Var) else int(d) for d in dims]
    dense = np.zeros(dims, dtype=np.float32)
    np.add.at(dense, tuple(np.asarray(idx.val).astype(np.int64)), np.asarray(vals.val, dtype=np.float32))
    print("SPARSE-TO-DENSE", dims, "nonzeros", np.asarray(vals.val).size)
    context.add(mb.const(val=dense, name=node.name))

_TORCH_OPS_REGISTRY.set_func_by_name(_sparse_coo_op, "sparse_coo_tensor")

# --- patch 19: Python-level checks (x is None, not x, item in list), decided at conversion time ---
def _bool_const(context, node, value):
    context.add(mb.const(val=bool(value), name=node.name))

def _is_op(context, node):
    a, b = _tops._get_inputs(context, node, expected=2)
    _bool_const(context, node, a is b)

def _isnot_op(context, node):
    a, b = _tops._get_inputs(context, node, expected=2)
    _bool_const(context, node, a is not b)

def _not_op(context, node):
    a = _tops._get_inputs(context, node, expected=1)[0]
    _bool_const(context, node, not (a.val if isinstance(a, Var) else a))

def _contains_op(context, node):
    box, item = _tops._get_inputs(context, node, expected=2)
    box = box.val if isinstance(box, Var) else box
    box = box.tolist() if hasattr(box, "tolist") else box
    item = item.val if isinstance(item, Var) else item
    _bool_const(context, node, item in box)

for _name, _fn in (("is", _is_op), ("isnot", _isnot_op), ("not", _not_op), ("contains", _contains_op)):
    _TORCH_OPS_REGISTRY.set_func_by_name(_fn, _name)


# --- patch 20: an empty list (e.g. annotate(List[Tensor], [])) must be a Python list, not an empty constant tensor ---
_orig_listconstruct = _tops.listconstruct

def _listconstruct_op(context, node):
    if len(node.inputs) == 0:
        context.add([], node.name)
        return
    return _orig_listconstruct(context, node)

_TORCH_OPS_REGISTRY.set_func_by_name(_listconstruct_op, "listconstruct")

# --- patch 22: x.to(y.dtype): coremltools turns y.dtype into a string like "fp32", but its `to` expects a torch dtype number ---
_orig_to = _tops.to

@register_torch_op(torch_alias=["_to_copy"], override=True)
def to(context, node):
    inputs = _tops._get_inputs(context, node)
    pos = {4: 1, 5: 1, 7: 1, 8: 1, 6: 2}.get(len(inputs))  # where the dtype sits, same rule as coremltools' `to`
    dt = inputs[pos] if pos is not None else None
    if isinstance(dt, Var) and isinstance(dt.val, str):  # dtype came from prim::dtype as a string
        context.add(_tops._cast_to(inputs[0], dt.val, node.name))  # cast with the string directly (fp64 falls back to fp32)
        return
    return _orig_to(context, node)  # every other `to` keeps the original behavior

# --- patch 23: prim::dtype gives the torch dtype number (6 = float32) instead of a string like "fp32", so zeros/empty/to can look it up ---
_DTYPE_STR_TO_NUM = {"uint8": 0, "int8": 1, "int16": 2, "int32": 3, "int64": 4, "fp16": 5, "fp32": 6, "fp64": 7, "bool": 11}

@register_torch_op(override=True)
def dtype(context, node):
    x = _tops._get_inputs(context, node, expected=1)[0]
    context.add(mb.const(val=_DTYPE_STR_TO_NUM[x.dtype.__name__], name=node.name))  # same lookup as coremltools, but returns the number

# --- patch 24: x[..., i, j] = v: coremltools' assignment pass drops each select's dim and applies the indices to dims 0, 1, ... in order ---
import coremltools.converters.mil.frontend.torch.converter as _tconv
_orig_assign_pass = _tconv.generate_tensor_assignment_ops

def _assign_pass(graph):
    producer = {o: n for n in graph.nodes for o in n.outputs}  # output name -> node that makes it
    dims_at = {}
    for i, n in enumerate(graph.nodes):
        if n.kind == "copy_":
            dims, src = [], producer.get(n.inputs[0])
            while src is not None and src.kind in ("select", "slice"):  # walk back the chain of views
                dims.append((src.kind, src.inputs[1]))  # (kind, name of its dim constant)
                src = producer.get(src.inputs[0])
            dims_at[i] = dims[::-1]  # first view first
    _orig_assign_pass(graph)
    for i, dims in dims_at.items():
        graph.nodes[i].assign_dims = dims  # the pass replaces node i in place; keep the real dims on it

_tconv.generate_tensor_assignment_ops = _assign_pass  # the converter looks the pass up by this name

_orig_inplace_copy = _tops._internal_op_tensor_inplace_copy

def _inplace_copy_op(context, node):
    dims = getattr(node, "assign_dims", None)
    if not dims or any(k != "select" for k, _ in dims):  # only plain index chains are remapped
        return _orig_inplace_copy(context, node)
    data = context[node.inputs[0]]
    left = list(range(data.rank))  # original dims still present after each select
    picked = [left.pop(int(context[d].val) % len(left)) for _, d in dims]  # each dim counts on the already-reduced tensor
    if picked == list(range(len(picked))):  # already dims 0, 1, ...: the original is right
        return _orig_inplace_copy(context, node)
    perm = picked + left  # selected dims first, in select order
    context.add(mb.transpose(x=data, perm=perm), node.name + "_perm_in")
    name, src = node.name, node.inputs[0]
    node.inputs[0], node.name = name + "_perm_in", name + "_perm_out"  # run the original on the moved tensor
    try:
        _orig_inplace_copy(context, node)
    finally:
        node.inputs[0], node.name = src, name  # restore, in case an unrolled loop converts this node again
    back = [int(i) for i in np.argsort(perm)]  # inverse permutation
    context.add(mb.transpose(x=context[name + "_perm_out"], perm=back, name=name))

_TORCH_OPS_REGISTRY.set_func_by_name(_inplace_copy_op, "_internal_op_tensor_inplace_copy")

# --- patch 25: tuple unpack of a module attribute that arrives as a str (SparseLinear.sparse_shape), same idea as patch 17 ---
_orig_tupleunpack = _tops.tupleunpack

def _tupleunpack_op(context, node):
    values = _tops._get_inputs(context, node, expected=1)[0]
    if isinstance(values, str):
        print("TUPLEUNPACK-STR", repr(values)[:120])
        try:
            values = ast.literal_eval(values)  # the tuple written out as text
        except Exception:  # not readable: take the real shape from the loaded model (the only str tuple seen so far)
            values = getattr(model.head_pose.mhr.pose_correctives_model.pose_dirs_predictor, "0").sparse_shape
        if len(values) != len(node.outputs):
            raise ValueError(f"TUPLEUNPACK-STR expected {len(node.outputs)} values, got {values}")
        for v, out in zip(values, node.outputs):
            context.add(mb.const(val=np.int32(v), name=out))  # each item becomes an int constant
        return
    return _orig_tupleunpack(context, node)

_TORCH_OPS_REGISTRY.set_func_by_name(_tupleunpack_op, "tupleunpack")

# --- patch 26: == where one side is an int module attribute that arrives as its name (e.g. 'num_vertices.1') ---
def _attr_value(s):  # look the attribute up by name on the loaded rig
    name = s.rsplit(".", 1)[0]  # drop the ".1" suffix
    vals = {repr(getattr(m, name)): getattr(m, name) for _, m in model.head_pose.mhr.named_modules() if hasattr(m, name)}
    if len(vals) != 1:  # missing, or several modules with different values: stop instead of guessing
        raise ValueError(f"ATTR-STR {s!r}: found {len(vals)} different values")
    return next(iter(vals.values()))

_orig_eq = _tops.eq

def _eq_op(context, node):
    x, y = _tops._get_inputs(context, node, expected=2)
    if not (isinstance(x, str) or isinstance(y, str)):
        return _orig_eq(context, node)  # every other == keeps the original behavior
    x, y = [_attr_value(v) if isinstance(v, str) else v for v in (x, y)]
    xv = x.val if isinstance(x, Var) else x  # value known at conversion time, or None
    yv = y.val if isinstance(y, Var) else y
    print("EQ-STR", xv, yv)
    if xv is not None and yv is not None:  # both known: the answer is a constant, so patch 15 can pick the branch
        context.add(mb.const(val=bool(np.all(np.equal(xv, yv))), name=node.name))
    else:  # one side known only at run time: compare as int32
        var, num = (x, yv) if isinstance(x, Var) else (y, xv)
        context.add(mb.equal(x=var, y=np.int32(num), name=node.name))

_TORCH_OPS_REGISTRY.set_func_by_name(_eq_op, "eq")

# --- patch 27: x[:, idx] = v writes into a ':' slice (a view) of x, but later ops read x itself; coremltools converts
#     index_put as a plain function, so the write is lost. Rewrite the traced graph: write into x, later reads use the result ---
_INT_MAX = 2**63 - 1  # end value of a ':' slice in the traced graph
_DYN = object()  # marks a slice bound that is computed, not constant
_VIEW_OPS = ("aten::slice", "aten::select", "aten::view", "aten::expand", "aten::narrow", "aten::as_strided", "aten::transpose", "aten::permute")
_WRITE_OPS = ("aten::index_put_", "aten::index_copy_")

def _const(v):  # value of a constant graph input, or _DYN if computed
    return v.toIValue() if v.node().kind() == "prim::Constant" else _DYN

def _is_full_slice(node):  # aten::slice(x, dim, 0, INT_MAX, 1) is the ':' in x[:, idx]: same memory as x
    if node.kind() != "aten::slice":
        return False
    start, end, step = [_const(node.inputsAt(i)) for i in (2, 3, 4)]
    return start in (0, None) and (end is None or (end is not _DYN and end >= _INT_MAX)) and step == 1

def _count_writes(block):  # write ops at any depth, to report how many sit inside if/loop blocks
    c = 0
    for n in block.nodes():
        c += n.kind() in _WRITE_OPS
        for b in n.blocks():
            c += _count_writes(b)
    return c

def _reroute_alias_writes(graph):
    done, skipped, top = 0, 0, 0
    for n in list(graph.nodes()):  # top level only: rerouting out of an if/loop block would break the graph's scoping
        if n.kind() not in _WRITE_OPS:
            continue
        top += 1
        base, moved = n.inputsAt(0), False
        while _is_full_slice(base.node()):  # follow ':' views back to the real tensor
            base, moved = base.node().inputsAt(0), True
        if base.node().kind() in _VIEW_OPS:  # writing into a partial view of another tensor: not handled, report it
            print("ALIAS-WRITES skipped (partial view)", n.kind(), base.node().kind(), str(n).strip()[:150])
            skipped += 1
            continue
        if moved:
            n.replaceInput(0, base)  # write into the real tensor, not the ':' view
        base.replaceAllUsesAfterNodeWith(n, n.output())  # later reads of the real tensor now see the write
        done += 1
    print("ALIAS-WRITES rerouted", done, "skipped", skipped, "inside if/loop blocks (not touched)", _count_writes(graph.block()) - top)

# Patches 29 and 33 are adapted from RoMa (https://github.com/naver/roma), Copyright (c) 2020 NAVER Corp.,
# BSD-3-Clause license (see RoMa's LICENSE; RoMa in turn adapts these conversions from SciPy, BSD-3-Clause).
# --- patch 29: roma.rotmat_to_unitquat picks its formula with nonzero() and writes quat[ind, k] = ... into views; the trace bakes in
#     the branch of the traced input and coremltools loses the view writes. Same math, all 4 candidates, chosen by mask ---
import roma

def _rotmat_to_unitquat(R):  # XYZW, same result as roma's (adapted from SciPy)
    m = R.reshape(-1, 3, 3)
    d = torch.diagonal(m, dim1=1, dim2=2)  # (N, 3): m00, m11, m22
    tr = d.sum(-1)  # trace
    m00, m11, m22 = d[:, 0], d[:, 1], d[:, 2]
    a = lambda r, c: m[:, r, c]  # matrix entry, shorthand
    cands = torch.stack([  # one candidate per largest choice: x, y, z, w (roma's i, j, k formulas written out)
        torch.stack([1 - tr + 2 * m00, a(1, 0) + a(0, 1), a(2, 0) + a(0, 2), a(2, 1) - a(1, 2)], -1),
        torch.stack([a(0, 1) + a(1, 0), 1 - tr + 2 * m11, a(2, 1) + a(1, 2), a(0, 2) - a(2, 0)], -1),
        torch.stack([a(0, 2) + a(2, 0), a(1, 2) + a(2, 1), 1 - tr + 2 * m22, a(1, 0) - a(0, 1)], -1),
        torch.stack([a(2, 1) - a(1, 2), a(0, 2) - a(2, 0), a(1, 0) - a(0, 1), 1 + tr], -1),
    ], 1)  # (N, 4 candidates, 4 components)
    choice = torch.argmax(torch.cat([d, tr[:, None]], -1), -1)  # roma's decision_matrix.argmax: first max wins
    mask = (torch.arange(4)[None, :] == choice[:, None]).to(m.dtype)  # (N, 4) one-hot of the chosen candidate
    q = (cands * mask[:, :, None]).sum(1)
    q = q / torch.norm(q, dim=1, keepdim=True)
    return q.reshape(R.shape[:-2] + (4,))

roma.rotmat_to_unitquat = _rotmat_to_unitquat  # roma.rotmat_to_euler calls it through this name (euler.py:258)
roma.mappings.rotmat_to_unitquat = _rotmat_to_unitquat  # and roma's own internal callers through this one

# --- patch 31: in-place arithmetic (x[:, 1:, 0] -= 1 in the rig's pose-corrective features) is lost too: coremltools converts sub_
#     as plain sub. Into a view: rewrite as view.copy_(view - c), which coremltools' assignment pass writes into the base tensor.
#     On a plain tensor: patch 27 reroutes later reads to the result (added to _WRITE_OPS below) ---
_ARITH_INPLACE = {"aten::add_": "aten::add", "aten::sub_": "aten::sub", "aten::mul_": "aten::mul", "aten::div_": "aten::div"}
_WRITE_OPS = _WRITE_OPS + tuple(_ARITH_INPLACE)  # patch 27's rerouting now also covers in-place arithmetic on plain tensors

def _split_inplace_arith(graph):
    done = 0
    for n in list(graph.nodes()):  # top level only, like patch 27
        if n.kind() not in _ARITH_INPLACE or n.inputsAt(0).node().kind() not in ("aten::select", "aten::slice"):
            continue
        view = n.inputsAt(0)
        f = graph.create(_ARITH_INPLACE[n.kind()], list(n.inputs()))  # same op and inputs, but returns a new tensor
        f.output().setType(n.output().type())
        f.insertBefore(n)
        no = graph.create("prim::Constant")  # copy_'s non_blocking=False argument
        no.i_("value", 0)
        no.output().setType(torch._C.BoolType.get())
        no.insertBefore(n)
        cp = graph.create("aten::copy_", [view, f.output(), no.output()])  # write the result back into the view
        cp.output().setType(view.type())
        cp.insertBefore(n)
        n.output().replaceAllUsesWith(f.output())  # later readers get the new slice values (what the in-place op returned), not the whole tensor
        n.destroy()
        done += 1
    print("INPLACE-ARITH split", done)

# --- patch 32: result = torch.empty(...).to(device), then result[..., 0, 0] = v (the rig's batch6DFromXYZ): coremltools' assignment
#     pass writes the first copy into the empty tensor's name, but later reads use the name after .to(), so that write is lost
#     (pose feature column 0 came out as 0 - 1 = -1). A .to() that keeps the dtype returns the same tensor in PyTorch, so drop it ---
def _drop_noop_to(graph):
    done = 0
    for n in list(graph.nodes()):  # top level only, like the assignment pass
        if n.kind() != "aten::to":
            continue
        ins = list(n.inputs())
        if len(ins) < 3 or ins[1].type().str() != "Device" or ins[2].type().str() != "NoneType":
            continue  # only the device-only form to(device, dtype=None, ...); dtype changes like to(x, 7) are kept
        if any(v.type().str() == "bool" and v.node().kind() == "prim::Constant" and v.toIValue() for v in ins[3:]):
            continue  # non_blocking=True or copy=True: not a plain alias, keep
        n.output().replaceAllUsesWith(ins[0])  # one device: PyTorch returns the input tensor itself
        n.destroy()
        done += 1
    print("NOOP-TO dropped", done)

# (adapted from RoMa's unitquat_to_euler, Copyright (c) 2020 NAVER Corp., BSD-3-Clause; see the note above patch 29)
# --- patch 33: roma.unitquat_to_euler (used by rotmat_to_euler for global_rot) fixes singular cases with angles[k][mask] = ... and
#     wraps angles with foo[foo < -pi] += 2*pi: the trace bakes in which elements the trace input changed, so the converted model
#     only fits inputs like the traced one (other inputs fail on the CPU). Same steps, every masked write as torch.where ---
import math

def _unitquat_to_euler(convention, quat, as_tuple=False, degrees=False, epsilon=1e-7):
    assert len(convention) == 3
    pi, lamb = math.pi, math.pi / 2
    extrinsic = convention.islower()
    if not extrinsic:
        convention = convention.lower()[::-1]
    quat, batch_shape = roma.internal.flatten_batch_dims(quat, end_dim=-2)
    i, j, k = (roma.euler._elementary_basis_index(c) for c in convention)
    assert i != j and j != k, "Consecutive axes should not be identical."
    symmetric = i == k
    if symmetric:
        k = 3 - i - j
    sign = (i - j) * (j - k) * (k - i) // 2  # +1 for an even permutation, -1 for odd
    if symmetric:
        a, b, c, d = quat[:, 3], quat[:, i], quat[:, j], quat[:, k] * sign
    else:
        a, b = quat[:, 3] - quat[:, j], quat[:, i] + quat[:, k] * sign
        c, d = quat[:, j] + quat[:, 3], quat[:, k] * sign - quat[:, i]
    angle_first, angle_third = (0, 2) if extrinsic else (2, 0)
    angles = [None, None, None]
    angles[1] = 2 * torch.atan2(torch.hypot(c, d), torch.hypot(a, b))
    case1 = torch.abs(angles[1]) <= epsilon  # singular: second angle is 0
    case2 = torch.abs(angles[1] - pi) <= epsilon  # singular: second angle is pi
    half_sum, half_diff = torch.atan2(b, a), torch.atan2(d, c)
    angles[angle_first] = half_sum - half_diff
    angles[angle_third] = half_sum + half_diff
    angles[2] = torch.where(torch.logical_or(case1, case2), torch.zeros_like(angles[2]), angles[2])  # was angles[2][case1or2] = 0
    angles[0] = torch.where(case1, 2 * half_sum, angles[0])  # was angles[0][case1] = 2 * half_sum[case1]
    angles[0] = torch.where(case2, 2 * (-1 if extrinsic else 1) * half_diff, angles[0])  # was angles[0][case2] = ...
    if not symmetric:
        angles[angle_third] = angles[angle_third] * sign
        angles[1] = angles[1] - lamb
    for idx in range(3):
        foo = angles[idx]
        foo = torch.where(foo < -pi, foo + 2 * pi, foo)  # was foo[foo < -pi] += 2 * pi
        foo = torch.where(foo > pi, foo - 2 * pi, foo)  # was foo[foo > pi] -= 2 * pi
        if degrees:
            foo = torch.rad2deg(foo)
        angles[idx] = roma.internal.unflatten_batch_dims(foo, batch_shape)
    return tuple(angles) if as_tuple else torch.stack(angles, dim=-1)

roma.euler.unitquat_to_euler = _unitquat_to_euler  # rotmat_to_euler calls it through this name (euler.py:257)
roma.unitquat_to_euler = _unitquat_to_euler  # and other callers through the package name


    # --- patch 7: Meta's batchXYZfrom6D assigns out_euler[..., k] in place; rebuild it with torch.stack instead ---
    # (same math as Meta's batchXYZfrom6D in sam-3d-body, sam_3d_body/models/modules/mhr_utils.py, rewritten without in-place writes)
import torch.nn.functional as F
import sam_3d_body.models.modules.mhr_utils as _mu

def _batchXYZfrom6D(poses):  # same math as Meta's version, no in-place assignment
    x_raw, y_raw = poses[..., :3], poses[..., 3:]  # first and second columns of the rotation
    cx = F.normalize(x_raw, dim=-1)
    cz = F.normalize(torch.cross(cx, y_raw, dim=-1), dim=-1)
    cy = torch.cross(cz, cx, dim=-1)
    m = torch.stack([cx, cy, cz], dim=-1)  # rotation matrix: ... x 3 x 3
    sy = torch.sqrt(m[..., 0, 0] * m[..., 0, 0] + m[..., 1, 0] * m[..., 1, 0])
    singular = (sy < 1e-6).float()  # 1 where the angles are degenerate
    x = torch.atan2(m[..., 2, 1], m[..., 2, 2])
    y = torch.atan2(-m[..., 2, 0], sy)
    z = torch.atan2(m[..., 1, 0], m[..., 0, 0])
    xs = torch.atan2(-m[..., 1, 2], m[..., 1, 1])
    ys = torch.atan2(-m[..., 2, 0], sy)
    zs = m[..., 1, 0] * 0
    return torch.stack([x * (1 - singular) + xs * singular,
                        y * (1 - singular) + ys * singular,
                        z * (1 - singular) + zs * singular], dim=-1)

_mu.batchXYZfrom6D = _batchXYZfrom6D  # the other functions in mhr_utils look it up by name, so they pick up the new one

# --- load the model ---
import sam_3d_body

base = os.environ.get("SAM3D_CKPT_DIR", "checkpoints/sam-3d-body-dinov3")
out = sam_3d_body.load_sam_3d_body(checkpoint_path=f"{base}/model.ckpt", device="cpu", mhr_path=f"{base}/assets/mhr_model.pt")
model = out[0] if isinstance(out, tuple) else out
model.eval()

class HeadWrapper(torch.nn.Module):  # tracing needs one plain tensor out
    def __init__(self, h):
        super().__init__()
        self.h = h
    def forward(self, token, init):
        return self.h(token, init)["pred_keypoints_3d"]  # (1, 70, 3)

wrapped = HeadWrapper(model.head_pose.float()).eval()
token = torch.randn(1, 1024)  # shapes taken from the trace
init = torch.randn(1, 519)
with torch.no_grad():
    ref = wrapped(token, init)
    print("keypoints shape:", ref.shape)

traced = torch.jit.trace(wrapped, (token, init), strict=False, check_trace=False)
torch._C._jit_pass_inline(traced.forward.graph)  # flatten submodule calls, so each write and its ':' view are in one graph
_split_inplace_arith(traced.forward.graph)  # patch 31, before patch 27's rerouting
_drop_noop_to(traced.forward.graph)  # patch 32
_reroute_alias_writes(traced.forward.graph)  # patch 27

# --- scan: list ops in the live graph that have no converter ---
g = traced.graph
torch._C._jit_pass_inline(g)  # flatten the rig's code so every op is visible
torch._C._jit_pass_dce(g)  # drop unused branches, so the list shows only ops the output needs

def _kinds(block):  # every op in the graph, including inside nested blocks
    ks = []
    for n in block.nodes():
        ks.append(n.kind())
        for b in n.blocks():
            ks += _kinds(b)
    return ks

missing = sorted({k for k in _kinds(g) if _TORCH_OPS_REGISTRY.get_func(k) is None
                  and k not in ("prim::GetAttr", "prim::SetAttr", "prim::CallMethod")})
print("unsupported ops:", missing)

# --- convert ---
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="token", shape=(1, 1024)), ct.TensorType(name="init", shape=(1, 519))],
    outputs=[ct.TensorType(name="keypoints3d")],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
)
mlmodel.save("sam3d_head_pose.mlpackage")
print("saved")

try:  # compare Core ML against PyTorch on the same input
    res = mlmodel.predict({"token": token.numpy(), "init": init.numpy()})
    print("max abs diff:", np.abs(ref.numpy() - list(res.values())[0]).max())
except Exception as e:
    print("predict failed:", e)
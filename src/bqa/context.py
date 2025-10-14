from functools import reduce

def run_pipeline(init_arg, *funcs) : return reduce(lambda res, func: func(res), funcs, init_arg)

def for_each(func, *it):
        for elems in zip(*it):
                func(*elems)

def append_fn(lst, elem):
        lst.append(elem)

def extend_by_defaults(lst, size, make_default_fn):
        while len(lst) < size:
                lst.append(make_default_fn())

def get_minimal_required_nodes_number(*ids):
        ids_num = len(ids)
        if ids_num == 1:
                return ids[0] + 1
        elif ids_num > 1:
                return max(ids[0], ids[1], *ids[2:]) + 1
        else:
                raise AssertionError(ids)

def assign_msg_to_edge(edge_to_msg, lhs_id, rhs_id):
        assert edge_to_msg.get((lhs_id, rhs_id)) is None
        assert edge_to_msg.get((rhs_id, lhs_id)) is None
        edge_to_msg[(lhs_id, rhs_id)] = len(edge_to_msg)
        edge_to_msg[(rhs_id, lhs_id)] = len(edge_to_msg)

def assign_lmbd_to_edge(edge_to_lmbd, lhs_id, rhs_id):
        assert edge_to_lmbd.get((lhs_id, rhs_id)) is None
        assert edge_to_lmbd.get((rhs_id, lhs_id)) is None
        new_lmbd_position = len(edge_to_lmbd) // 2
        edge_to_lmbd[(lhs_id, rhs_id)] = new_lmbd_position
        edge_to_lmbd[(rhs_id, lhs_id)] = new_lmbd_position

def assign_smthng_to_edge(edge_to_smthng, lhs_id, rhs_id, smthng):
        assert edge_to_smthng.get((lhs_id, rhs_id)) is None
        assert edge_to_smthng.get((rhs_id, lhs_id)) is None
        edge_to_smthng[(lhs_id, rhs_id)] = smthng
        edge_to_smthng[(rhs_id, lhs_id)] = smthng

def make_default_node_content() : return []

def insert_edge_to_graph(graph, lhs_id, rhs_id):
        extend_by_defaults(graph, get_minimal_required_nodes_number(lhs_id, rhs_id), make_default_node_content)
        graph[lhs_id].append(rhs_id)
        graph[rhs_id].append(lhs_id)

def extend_nodes_by_defaults(nodes, lhs_id, rhs_id):
        extend_by_defaults(nodes, get_minimal_required_nodes_number(lhs_id, rhs_id), lambda : 0)

def make_context(backend):
        return {"graph"      : [],
                "edge->ampl" : {},
                "node->ampl" : [],
                "edge->msg"  : {},
                "edge->lmbd" : {},
                "msg->lmbd"  : [], 
                "layout"     : {},
                "backend"    : backend}

def get_backend(context) : return context["backend"]

def get_qubits_number(context) : return len(context["node->ampl"])

def get_msgs_number(context) : return len(context["edge->msg"])

def get_lmbds_number(context) : return get_msgs_number(context) // 2

def get_degree_layout(context) : return context["layout"]

def insert_edge(context, edge_record):
        (lhs_id, rhs_id), ampl = edge_record
        assert lhs_id < rhs_id
        insert_edge_to_graph(context["graph"], lhs_id, rhs_id)
        extend_nodes_by_defaults(context["node->ampl"], lhs_id, rhs_id)
        assign_smthng_to_edge(context["edge->ampl"], lhs_id, rhs_id, ampl)
        assign_msg_to_edge(context["edge->msg"], lhs_id, rhs_id)
        assign_lmbd_to_edge(context["edge->lmbd"], lhs_id, rhs_id)
        return context

def insert_node(context, node_record):
        node_id, ampl = node_record
        min_nodes_number = get_minimal_required_nodes_number(node_id)
        extend_by_defaults(context["graph"], min_nodes_number, make_default_node_content)
        node_to_ampl = context["node->ampl"]
        extend_by_defaults(node_to_ampl, min_nodes_number, lambda : 0)
        node_to_ampl[node_id] = ampl
        return context

def insert_empty_layout_per_degree(context, degree):
        layout = get_degree_layout(context)
        layout[degree] = {"node_id" : [],
                          "input_msgs" : [[] for _ in range(degree)],
                          "output_msgs" : [[] for _ in range(degree)],
                          "lmbds" : [[] for _ in range(degree)]}
        return layout[degree]

def turn_to_tensors(context):
        layout = get_degree_layout(context)
        backend = get_backend(context)
        new_layout = {k : {"node_id" : backend.from_list(v["node_id"]),
                           "input_msgs" : list(map(backend.from_list, v["input_msgs"])),
                           "output_msgs" : list(map(backend.from_list, v["output_msgs"])),
                           "lmbds" : list(map(backend.from_list, v["input_msgs"]))} for k, v in layout.items()}
        context["layout"] = new_layout
        context["msg->lmbd"] = backend.from_list(context["msg->lmbd"])
        return context

def get_or_create_degree_layout(context, degree):
        return get_degree_layout(context).get(degree) or insert_empty_layout_per_degree(context, degree)

def build_layout_per_node(context, node):
        node_id, neighbors = node
        edge_to_msg = context["edge->msg"]
        edge_to_lmbd = context["edge->lmbd"]
        degree = len(neighbors)
        degree_layout = get_or_create_degree_layout(context, degree)
        input_msgs = map(lambda src: edge_to_msg[(src, node_id)], neighbors)
        output_msgs = map(lambda dst: edge_to_msg[(node_id, dst)], neighbors)
        lmbds = map(lambda other_id: edge_to_lmbd[(node_id, other_id)], neighbors)
        degree_layout["node_id"].append(node_id)
        for_each(append_fn, degree_layout["input_msgs"], input_msgs)
        for_each(append_fn, degree_layout["output_msgs"], output_msgs)
        for_each(append_fn, degree_layout["lmbds"], lmbds)
        return context

def build_layout(context):
        graph = context["graph"]
        return reduce(build_layout_per_node, enumerate(graph), context)

def build_msg_to_lmbd(context):
        edge_to_msg, edge_to_lmbd = context["edge->msg"], context["edge->lmbd"]
        assert len(edge_to_msg) == len(edge_to_lmbd)
        def builder(acc, pair):
                edge, msg = pair
                acc[msg] = edge_to_lmbd[edge]
                return acc
        size = len(edge_to_msg)
        context["msg->lmbd"] = reduce(builder, edge_to_msg.items(), size * [0])
        return context

def build_context(spec, backend):

    def get_edges():
        for (lhs_id, rhs_id), ampl in filter(lambda key_val: isinstance(key_val[0], tuple), spec.items()):
                if (rhs_id, lhs_id) in spec:
                        raise ValueError(f"Edge connecting {lhs_id} and {rhs_id} is duplicated")
                yield (min(lhs_id, rhs_id), max(lhs_id, rhs_id)), ampl

    def get_nodes():
        return filter(lambda key_val: isinstance(key_val[0], int), spec.items())

    def insert_edges(context):
        return reduce(insert_edge, get_edges(), context)

    def insert_nodes(context):
        return reduce(insert_node, get_nodes(), context)

    return run_pipeline(
            make_context(backend),
            insert_edges,
            insert_nodes,
            build_layout,
            build_msg_to_lmbd,
            turn_to_tensors)


import logging
from bqa import run_qa
from bqa.exact_sim import run_qa_exact
from utils import get_trace_distance

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# the following config serves as DSL for quantum annealing task specification, it contains all the information necessary for simulation;
# most of fields in this config can be omitted, in this case warning is shown and the corresponding default value is substituted;
# if something is wrong, the internal checker will throw an error explaining where is the problem

config = {
    "edges" : { # ------------------------> this dict specifies interaction hamiltonian: its a map from edges to interaction amplitudes
        (0, 1) : 1., #                      edges must not be duplicated (eg (1, 0) and (0, 1) in a single dict causes error)
        (1, 2) : -1., #                     "edges" field is mandatory, "edges" field is flexible, it also can be a list or tuple of pairs
        (2, 3) : 1., #                      i.e. one can deserialize it directly from json
        (3, 4) : 1.,
        (0, 5) : -1.,
        (4, 6) : 1.,
        (5, 7) : -1.,
        (6, 11) : 1.,
        (7, 8) : 1.,
        (8, 9) : -1.,
        (9, 10) : -1.,
        (10, 11) : 1.,
        (11, 12) : -1.,
        (12, 13) : 1.,
        (9, 14) : -1.,
        (13, 15) : 1.,
        (14, 16) : 1.,
        (15, 20) : 1.,
        (16, 17) : -1.,
        (17, 18) : -1.,
        (18, 19) : -1.,
        (19, 20) : 1.,
    },
    "nodes" : { # ------------------------> this dict specifies local field acting on a node: its map from node to the amplitude of the field
        0 : -1., #                          this field is optional, default is the empty dict {}
        3 : -1., #                          if some node id is absent, the default value is taken from "default_field" (see below)
        6 : -1.,
        7 : -1.,
        9 : -1.,
        12 : -1.,
        13 : -1.,
        15 : -1,
        16 : -1,
        17 : -1,
        20 : -1.,
    },
    "default_field" : 1., # --------------> this is default field acting on the node, it is optional, default value is 0

    "max_bond_dim" : 8, # ----------------> this is maximal bond dimension allowed during simulation, it is optional, default value is 4

    "bp_eps" : 1e-10, # ------------------> this is the stopping discrepancy of BP algorithm, it is optional, default value is 1e-10

    "pinv_eps" : 1e-10, # ----------------> this is a pseudo inverse threshold, it is optional, default value is 1e-10

    "max_bp_iters_number" : 100, # -------> this is the maximal BP iterations allowed, if BP reaches this value,
                                 #          warning is shown and BP is terminated, it is optional, default value is 100

    "measurement_threshold" : 0.95, # ----> this is a confedence thresold which allows to skip measurement simulation
                                    #       and directly project state to the corresponding direction, it is optional, default value is 0.95

    "seed" : 42, # -----------------------> this is the random seed, it is optional, default value is 42

    "backend" : "numpy", # ---------------> this is a backend used for simulation, it is optional, default value is "numpy"
                         #                  other backends available are ["cupy"]

    "schedule" : { # ---------------------> this field specifies the annealing schedule, it is optional, to see default value see the corresponding warning

        "total_time" : 10., # ------------> total "physical" time of the annealing dynamics, it is optional, default value is 10

        "starting_mixing" : 1., # --------> starting mixing, i.e. simulation starts from
                                #           starting_mixing * H_mix + (1 - starting_mixing) * H_int hamiltonian, it is optional, default value is 1.

        "actions" : [ # ------------------> this is the list with actions (QA instructions), it is optional, to see the default value,
                      #                     see the corresponding warning message

            # QA instruction describing annealing dynamics:
            {
                "time" : 1., # -----------> the part of the "physical" time dedicated to this instruction

                "steps_number" : 100, # --> number of discrete time steps within this instruction

                "final_mixing" : 0.0, # --> the mixing after the instruction is finished, i.e. final hamiltonian is
                                      #     final_mixing * H_mix + (1 - final_mixing) * H_int hamiltonian
            },

            # QA instruction requesting to collect density matrices into a list
            "get_density_matrices",

            # QA instruction requesting to perform measurements of all nodes, results are collected as integer values into a list
            "measure",
        ]
    },
}

bptn_result = run_qa(config)  # run_qa runs BPTN based quantum annealing simulation
exact_result = run_qa_exact(config)  # run_qa_exact runs exact quantum annealing simulation
trace_distances = map(get_trace_distance, bptn_result[0], exact_result[0])

print("SIMULATION RESULTS:")

for node_id, trace_distance in enumerate(trace_distances):
    print(f"For node {node_id} trace distance between final exact and TNBP based density matrix is {trace_distance}")

print(f"Measurement results for BPTN based simulation: {bptn_result[1]}")
print(f"Measurement results for exact simulation:      {exact_result[1]}")

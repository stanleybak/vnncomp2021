'''
Fenerate vnnlib property files and acasxu_instances.csv for acasxu for vnncomp 2021

Stanley Bak, April 2021
'''

from math import pi

import numpy as np

def get_init_box(property_str):
    'get lb, ub lists for the given property'

    if property_str in ("1", "2"):
        init_lb = [55947.691, -pi, -pi, 1145, 0]
        init_ub = [60760, pi, pi, 1200, 60]
    elif property_str == "3":
        init_lb = [1500, -0.06, 3.1, 980, 960]
        init_ub = [1800, 0.06, pi, 1200, 1200]
    else:
        assert property_str == "4"
        init_lb = [1500, -0.06, 0, 1000, 700]
        init_ub = [1800, 0.06, 0, 1200, 800]
        
    return init_lb, init_ub

def get_spec(property_str):
    'get the specification mat and rhs'

    #labels = ['Clear of Conflict (COC)', 'Weak Left', 'Weak Right', 'Strong Left', 'Strong Right']

    if property_str == "1":
        desc = 'Unsafe if COC >= 1500. Output scaling is 373.94992 with a bias of 7.518884: (1500 - 7.518884) / 373.94992 = 3.991125'
        output_scaling_mean = 7.5188840201005975
        output_scaling_range = 373.94992
        
        # (1500 - 7.518884) / 373.94992 = 3.991125
        threshold = (1500 - output_scaling_mean) / output_scaling_range
        spec = (desc, [[-1, 0, 0, 0, 0]], [-threshold])
        
    elif property_str == "2":
        desc = 'Unsafe if COC is maximal'
        # y0 > y1 and y0 > y1 and y0 > y2 and y0 > y3 and y0 > y4
        spec = (desc, [[-1, 1, 0, 0, 0],
                 [-1, 0, 1, 0, 0],
                 [-1, 0, 0, 1, 0],
                 [-1, 0, 0, 0, 1]], [0, 0, 0, 0])
        
    elif property_str in ("3", "4"):
        desc = 'Unsafe if COC is minimal'
        spec = (desc, [[1, -1, 0, 0, 0],
                 [1, 0, -1, 0, 0],
                 [1, 0, 0, -1, 0],
                 [1, 0, 0, 0, -1]], [0, 0, 0, 0])

    return spec

def print_prop(num, f):
    'print prop to file f'

    init_lb, init_ub = get_init_box(num)
    
    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    f.write(f'; ACAS Xu property {num}\n')

    for i in range(len(init_lb)):
        f.write(f"\n; Unscaled Input {i}: {init_lb[i], init_ub[i]}\n")
        lb = (init_lb[i] - means_for_scaling[i]) / range_for_scaling[i]
        ub = (init_ub[i] - means_for_scaling[i]) / range_for_scaling[i]
        f.write(f"(assert (<= X_0_0_0_{i} {ub}))\n")
        f.write(f"(assert (>= X_0_0_0_{i} {lb}))\n")

    f.write("\n\n")

    # print spec
    desc, mat, rhs_vec = get_spec(num)

    f.write(f'; {desc}\n')

    for row, rhs in zip(mat, rhs_vec):
        row = np.array(row, dtype=float)

        if rhs == 0:
            assert sum(row != 0) == 2
            i1, i2 = np.where(row)[0]

            if row[i1] == 1.0 and row[i2] == -1.0:
                f.write(f"(assert (<= Y_0_{i1} Y_0_{i2}))\n")
            else:
                assert row[i1] == -1.0 and row[i2] == 1.0
                f.write(f"(assert (<= Y_0_{i2} Y_0_{i1}))\n")
        else:
            assert sum(row != 0) == 1

            i = np.argmax(np.abs(row))

            if row[i] > 0:
                f.write(f"(assert (<= Y_0_{i} {rhs}))\n")
            else:
                f.write(f"(assert (>= Y_0_{i} {-rhs}))\n")

def main():
    'main entry point'

    # generate spec files
    for spec in ["1", "2", "3", "4"]:
        filename = f"prop_{spec}.vnnlib"
        print(f"Generating {filename}...")
        
        with open(filename, 'w') as f:
            print_prop(spec, f)

    # generate csv
    total_timeout = 6 * 60 * 60 # 6 hours
    benchmark_timeout = total_timeout / (4 * 5 * 9)
    filename = "acasxu_instances.csv"
    
    with open(filename, 'w') as f:
        print(f"Generating {filename} with benchmark timeout {benchmark_timeout}...")
        
        for prop_num in range(1, 5):
            p = f'prop_{prop_num}.vnnlib'
            for a in range(1, 6):
                for b in range(1, 10):
                    o = f'ACASXU_run2a_{a}_{b}_batch_2000.onnx'

                    f.write(f'{o},{p},{benchmark_timeout}\n')


if __name__ == "__main__":
    main()

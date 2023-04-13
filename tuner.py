from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from utilities import read_test_log_file, stp_markham_dir, stp_tsodyks_dir, stp_moraitis_dir
import simulation
import brian2 as b2

base_args = {"test_mode": True, "clobber": True, "debug": True, "output": "./runs/", "stp_on": True}
search_space_markham = [Integer(100, 500, name="taud_msec"),
                        Integer(1, 5, name="tauf_msec"),
                        Real(0.2, 0.8, name="U")]
search_space_tsodyks = [Real(0.5, 10, name="Omega_d_sec"),
                        Real(0.5, 10, name="Omega_f_sec"),
                        Real(0.2, 0.8, name="U_0")]
search_space_moraitis = [Real(0.5, 10, name="tc_pre_ee_msec"),
                        Real(0.5, 10, name="tc_post_1_ee_msec"),
                        Real(0.2, 0.8, name="tc_post_2_ee_msec"),
                        Integer(50, 500, name="tc_lambda_msec")]

search_space = search_space_tsodyks

@use_named_args(search_space)
def objective(**params):
    return -get_accuracy_from_command(params)

def get_accuracy_from_command(params):
    # replace tagged parameters with correct brian2 units
    add_brian2_units(params)

    # combine all arguments
    base_args["custom_namespace"] = params

    # run simulation with args
    simulation.main(**base_args)

    # get final accuracy score from logs
    accuracy_dict = read_test_log_file(stp_tsodyks_dir).pop()
    accuracy = accuracy_dict["accuracy"]

    print(accuracy)
    return accuracy

def add_brian2_units(params):
    for key in params:
        if "_msec" in key or "_sec" in key:
            # key value from dict
            val = params[key]

            # get new value with correct unit, new key without tag
            if "_msec" in key:
                new_val = val * b2.ms
                new_key = key.replace("_msec", "")
            else:
                new_val = val / b2.second
                new_key = key.replace("_sec", "")

            # remove old entry, add new entry
            del params[key]
            params[new_key] = new_val

def tuning_pipeline(rule_name):
    # set additional arguments
    additional_args = {"runname": "output_stp_%s" % rule_name, "stp_rule": rule_name}
    base_args.update(additional_args)

    gaussian_process = gp_minimize(objective, search_space, n_calls=25, random_state=0)

    dump(gaussian_process, rule_name + ".pk1")

    # display optimal parameters with respective accuracy
    for param in gaussian_process.x:
        print(param)
    print(-gaussian_process.fun)

tuning_pipeline("tsodyks")

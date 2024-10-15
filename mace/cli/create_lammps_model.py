import os
import argparse
import warnings

import torch
from e3nn.util import jit

from mace.calculators import LAMMPS_MACE, LAMMPS_MACE_CHARGE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model to be converted to LAMMPS",
    )
    parser.add_argument(
        "--head",
        type=str,
        nargs="?",
        help="Head of the model to be converted to LAMMPS",
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="?",
        help="Data type of the model to be converted to LAMMPS",
        default="float64",
    )
    parser.add_argument(
        "--charge_cv_expr",
        type=str,
        nargs="?",
        help="Expression of the charge CV",
        default=None,
    )
    return parser.parse_args()


def select_head(model):
    if hasattr(model, "heads"):
        heads = model.heads
    else:
        heads = [None]

    if len(heads) == 1:
        print(f"Only one head found in the model: {heads[0]}. Skipping selection.")
        return heads[0]

    print("Available heads in the model:")
    for i, head in enumerate(heads):
        print(f"{i + 1}: {head}")

    # Ask the user to select a head
    selected = input(
        f"Select a head by number (Defaulting to head: {len(heads)}, press Enter to accept): "
    )

    if selected.isdigit() and 1 <= int(selected) <= len(heads):
        return heads[int(selected) - 1]
    if selected == "":
        print("No head selected. Proceeding without specifying a head.")
        return None
    print(f"No valid selection made. Defaulting to the last head: {heads[-1]}")
    return heads[-1]


def main():
    args = parse_args()
    model_path = args.model_path  # takes model name as command-line input
    model = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if args.dtype == "float64":
        model = model.double().to("cpu")
    elif args.dtype == "float32":
        print("Converting model to float32, this may cause loss of precision.")
        model = model.float().to("cpu")

    if args.head is None:
        head = select_head(model)
    else:
        head = args.head
        print(
            f"Selected head: {head} from command line in the list available heads: {model.heads}"
        )

    lammps_model = (
        LAMMPS_MACE(model, head=head) if head is not None else LAMMPS_MACE(model)
    )
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(model_path + "-lammps.pt")

    if args.charge_cv_expr is not None:
        model_name = model._get_name()
        if model_name != 'EnergyChargesMACE':
            warnings.warn(
                '{:s} is not a charge model!i '.format(args.model_path)
                + ' Will ignore the given charge CV expression!'
            )
        else:
            with open('charge_cv_expr.py', 'w') as fp:
                function = 'import torch\n@torch.jit.script\n'
                function += 'def function(c: torch.Tensor) -> torch.Tensor:\n'
                function += '    return {:s}\n'.format(args.charge_cv_expr)
                function += 'expr="{:s}"\n'.format(args.charge_cv_expr)
                print(function, file=fp)
            lammps_model = (
                LAMMPS_MACE_CHARGE(model, head=head) if head is not None else
                LAMMPS_MACE_CHARGE(model)
            )
            lammps_model_compiled = jit.compile(lammps_model)
            lammps_model_compiled.save(
                os.path.basename(model_path) + "-lammps_charge.pt"
            )
            os.remove('charge_cv_expr.py')


if __name__ == "__main__":
    main()

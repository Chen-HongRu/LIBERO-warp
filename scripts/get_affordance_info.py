# Print the affordance information specified in the registered object XML files.

from libero.libero.envs.objects import OBJECTS_DICT
from libero.libero.utils.object_utils import get_affordance_regions


def main():
    """Print affordance regions for every registered LIBERO object."""
    affordances = get_affordance_regions(OBJECTS_DICT)
    print(affordances)


if __name__ == "__main__":
    main()

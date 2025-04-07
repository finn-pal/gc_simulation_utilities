import os
import sys

import gizmo_analysis as gizmo
import halo_analysis as halo
import numpy as np
import pandas as pd
import utilities as ut
from tqdm import tqdm


def block_print():
    """
    Prevents prining of statements. Useful for when using halo tools as a lot of erroneous print statements.
    """

    sys.stdout = open(os.devnull, "w")


def enable_print():
    """
    Enables prints statements.
    """

    sys.stdout = sys.__stdout__


def get_halo_cid(halt, halo_tid: int, fire_dir: str) -> tuple[int, int]:
    """
    Finds the corresponding halo catalogue id (cid) from a provided halo id from the halo tree (tid).

    Args:
        halt (_type_): Halo tree
        halo_tid (int): Halo id from the halo tree
        fire_dir (str): Directory of the FIRE simulation data (of form "/m12i_res7100")

    Returns:
        tuple[int, int]: Returns the halo catalogue id (cid) and the corresponding snapshot
    """

    # get index of halo in halo tree
    idx = np.where(halt["tid"] == halo_tid)[0][0]
    # get the corresponding snapshot (halo catalogue) and index of the halo in the halo catalogue
    snap = halt["snapshot"][idx]
    halo_idx = halt["catalog.index"][idx]
    # import the relevant halo catalogue
    hal = halo.io.IO.read_catalogs("index", snap, simulation_directory=fire_dir, species=None)
    # get the halo catalogue id (cid)
    halo_cid = hal["id"][halo_idx]
    return halo_cid, snap


def main_prog_halt(halt, main_halo_tid) -> list[int]:
    """
    Get a list of the halo tree ids for the most massive progenitors of the main galaxy from gizmo halo tree

    Args:
        halt (_type_): Halo tree

    Returns:
        list[int]: List of halo tree halo ids (tid) tracing the main progenitors of the most massive galaxy at
        z = 0.
    """
    # main galaxy has index 0 in the halo tree
    main_halo_idx = np.where(halt["tid"] == main_halo_tid)[0][0]
    main_halo_lst = [main_halo_idx]

    # FIRE has 600 snapshots but progenitor has usally not formed much earlier than snapshot 10
    for _ in range(1, 590):
        idx = halt["progenitor.main.index"][main_halo_lst[-1]]
        main_halo_lst.append(idx)

    tid_main_lst = halt["tid"][main_halo_lst]

    return tid_main_lst


def naming_value(flag) -> int:
    """
    Converts flag value to interger for input into file naming for processed data

    Args:
        flag (_type_): Flag value (can be 0, 1, or None)

    Returns:
        int: Flag for file name (0, 1 or 2)
    """
    if flag is None:
        flag_name = 2
    else:
        flag_name = flag
    return flag_name


def get_descendants_halt(halo_tid: int, halt) -> list[int]:
    """
    Get list of descendents (tid's) of the halo in question

    Args:
        halo_tid (int): Halo id from the halo tree
        halt (_type_): Halo tree

    Returns:
        list[int]: A list of descendant halos (list of tid's)
    """
    halo_idx = np.where(halt["tid"] == halo_tid)[0][0]
    halo_snap = halt["snapshot"][halo_idx]
    desc_lst = [halo_idx]

    # get a list of all descendents of the halo up to z = 0
    for _ in range(halo_snap, 600):
        idx = halt["descendant.index"][desc_lst[-1]]
        desc_lst.append(idx)

    return desc_lst


def iteration_name(it: int) -> str:
    """
    Get 3 digit iteration id name from iteration number

    Args:
        it (int): Iteration number.

    Returns:
        str: Iteration id name.
    """
    # ensure group naming is consitent with three digits
    if len(str(it)) == 1:
        it_id = "it00" + str(it)
    elif len(str(it)) == 2:
        it_id = "it0" + str(it)
    else:
        it_id = "it" + str(it)

    return it_id


def get_halo_tree(sim: str, sim_dir: str, assign_hosts_rotation: bool = False):
    """
    Given directory to simulation returns halo tree.

    Args:
        sim (str): Simulation of interest (of form "m12i").
        sim_dir (str): Directory of the simulation data.

    Returns:
        _type_: Halo tree.
    """
    fire_dir = sim_dir + sim + "/" + sim + "_res7100/"

    for _ in tqdm(range(1), ncols=150, desc="Retrieving Halo Tree....................."):
        block_print()  # block verbose print statements
        halt = halo.io.IO.read_tree(
            simulation_directory=fire_dir, assign_hosts_rotation=assign_hosts_rotation
        )
        enable_print()

    return halt


def open_snapshot(
    snapshot: int, fire_dir: str, species: list[str] = ["all"], assign_hosts_rotation: bool = True
):
    """
    Get the particle details for a given snapshot. This contains all particles in publicly available snapshots
    with the coordinate frame to be centred on the central galaxy.

    Args:
        snapshot (int): Snapshot.
        fire_dir (str): Directory of the FIRE simulation data (of form "/m12i_res7100").

    Returns:
        _type_: The particle details for a given snapshot
    """

    for _ in tqdm(range(1), ncols=150, desc="Retrieving Snapshot %d.................." % snapshot):
        block_print()  # block verbose print statements
        part = gizmo.io.Read.read_snapshots(
            species, "index", snapshot, fire_dir, assign_hosts_rotation=assign_hosts_rotation
        )
        enable_print()
    return part


def snapshot_name(snap_num: int):
    """
    Get 3 digit iteration id name from snapshot.

    Args:
        snap_num (int): Snapshot.

    Returns:
        str: Snapshot id name.
    """
    # ensure group naming is consitent with three digits
    if len(str(snap_num)) == 1:
        snap_id = "snap00" + str(snap_num)
    elif len(str(snap_num)) == 2:
        snap_id = "snap0" + str(snap_num)
    else:
        snap_id = "snap" + str(snap_num)

    return snap_id


def get_main_vir_rad_snap(halt, main_halo_tid: int, snapshot: int) -> list[int]:
    """
    Get a list of the halo tree ids for the most massive progenitors of the main galaxy from gizmo halo tree

    Args:
        halt (_type_): Halo tree

    Returns:
        list[int]: List of halo tree halo ids (tid) tracing the main progenitors of the most massive galaxy at
        z = 0.
    """
    # main galaxy has index 0 in the halo tree
    main_halo_idx = np.where(halt["tid"] == main_halo_tid)[0][0]
    main_halo_lst = [main_halo_idx]

    # FIRE has 600 snapshots but progenitor has usally not formed much earlier than snapshot 10
    for _ in range(1, 590):
        idx = halt["progenitor.main.index"][main_halo_lst[-1]]
        main_halo_lst.append(idx)

    tid_main_lst = halt["tid"][main_halo_lst]
    snap_main_lst = halt["snapshot"][main_halo_lst]
    idx_interest = np.where(snap_main_lst == snapshot)[0][0]
    tid_interest = tid_main_lst[idx_interest]

    halt_idx_interest = np.where(halt["tid"] == tid_interest)[0][0]
    rad_interest = halt["radius"][halt_idx_interest]

    return rad_interest


def particle_type(quality: int) -> str:
    """
    Using the data quality flag from gc model return particle type. From the gc model:
    # 2: good (star with t_s > t_form - time_lag):
    # 1: half (star with t_s < t_form - time_lag)
    # 0: bad (dm particle)

    Args:
        quality (int): Quality flag output from gc formation model (assign function).

    Returns:
        str: Particle type. Either "star" or "dark".
    """
    if quality == 0:
        part = "dark"
    else:
        part = "star"
    return part


def get_different_snap_lst(main_halo_tid, halt, sim, sim_dir):
    snap_pub_dir = sim_dir + "/snapshot_times_public.txt"
    fire_dir = sim_dir + sim + "/" + sim + "_res7100/"

    snap_pub_data = pd.read_table(snap_pub_dir, comment="#", header=None, sep=r"\s+")
    snap_pub_data.columns = [
        "index",
        "scale_factor",
        "redshift",
        "time_Gyr",
        "lookback_time_Gyr",
        "time_width_Myr",
    ]

    tid_main_lst = main_prog_halt(halt, main_halo_tid)

    different_host_lst = []

    for snap in snap_pub_data["index"]:
        block_print()
        hal = halo.io.IO.read_catalogs("index", snap, fire_dir)
        enable_print()
        host_cat_idx = np.unique(hal["host.index"])[0]
        host_tre_idx = hal["tree.index"][host_cat_idx]
        host_tid = halt["tid"][host_tre_idx]

        if host_tid not in tid_main_lst:
            different_host_lst.append(snap)

        del hal

    return different_host_lst


def get_main_prog_at_snap(halt, main_halo_tid, snapshot):
    tid_main_lst = main_prog_halt(halt, main_halo_tid)
    halo_tid = tid_main_lst[600 - snapshot]

    # check halo_tid matches snapshot
    halo_idx = np.where(halt["tid"] == halo_tid)[0][0]
    if halt["snapshot"][halo_idx] != snapshot:
        print("Halo tid does not match at required snapshot")
        exit(1)  # Stops execution with a non-zero exit code

    return halo_tid


def get_halo_details(part, halt, halo_tid, snapshot):
    # do a check to ensure part matches intended snap
    if part.snapshot["index"] != snapshot:
        print("Part selection does not match intended snapshot for halo details")
        exit(1)  # Stops execution with a non-zero exit code

    halo_index = np.where(halt["tid"] == halo_tid)[0][0]

    center_pos = ut.particle.get_center_positions(
        part,
        species_name="star",
        center_positions=halt["position"][halo_index],
        return_single_array=False,
        verbose=False,
    )

    center_vel = ut.particle.get_center_velocities_or_accelerations(
        part,
        property_kind="velocity",
        species_name="star",
        part_indicess=None,
        weight_property="mass",
        distance_max=10,
        center_positions=center_pos,
        return_single_array=False,
        verbose=False,
    )

    center_rot = ut.particle.get_principal_axes(
        part,
        "star",
        distance_max=10,
        mass_percent=90,
        age_percent=25,
        center_positions=center_pos,
        center_velocities=center_vel,
        return_single_array=False,
        verbose=False,
    )

    halo_detail_dict = {
        "tid": halo_tid,
        "snapshot": snapshot,
        "position": center_pos,
        "velocity": center_vel,
        "rotation": center_rot["rotation"][0],
        "axis.ratios": center_rot["axis.ratios"],
    }

    return halo_detail_dict


def get_particle_halo_pos_vel(part, gc_id, ptype, halo_detail_dict, coordinates="cartesian"):
    # do a check to ensure part matches intended snap
    if part.snapshot["index"] != halo_detail_dict["snapshot"]:
        print("Part selection does not match intended snapshot for halo details")
        exit(1)  # Stops execution with a non-zero exit code

    part_idx = np.where(part[ptype]["id"] == gc_id)[0][0]

    dist_vect = ut.particle.get_distances_wrt_center(
        part,
        species=[ptype],
        part_indicess=np.array([part_idx]),
        center_position=halo_detail_dict["position"],
        rotation=halo_detail_dict["rotation"],
        coordinate_system=coordinates,
    )

    vel_vect = ut.particle.get_velocities_wrt_center(
        part,
        species=[ptype],
        part_indicess=np.array([part_idx]),
        center_position=halo_detail_dict["position"],
        center_velocity=halo_detail_dict["velocity"],
        rotation=halo_detail_dict["rotation"],
        coordinate_system=coordinates,
    )

    return dist_vect[0], vel_vect[0]


def get_correct_gc_part_idx(part, gc_id, gc_snapform, snapshot, ptype="star"):
    # this is a temporary solution. I do not know what to do if two possible particles have the same
    # formation snapshot

    # do a check to ensure part matches intended snap
    if part.snapshot["index"] != snapshot:
        print("Part selection does not match intended snapshot for halo details")
        exit(1)  # Stops execution with a non-zero exit code

    part_idxs = np.where(part[ptype]["id"] == gc_id)[0]
    part_snapform = part[ptype].prop("form.snapshot", part_idxs)

    idx = np.where(part_snapform == gc_snapform)[0]

    if len(idx) > 1:
        print("Multiple particles have same formation snapshot as GC, cannot determine correct GC particle")
        exit(1)  # Stops execution with a non-zero exit code

    if len(idx) == 0:
        print("No particles have same formation snapshot as GC, cannot determine correct GC particle")
        exit(1)  # Stops execution with a non-zero exit code

    correct_part_idx = part_idxs[idx][0]

    if part["star"]["id"][correct_part_idx] != gc_id:
        print("Error in determining correct GC particle index")
        exit(1)  # Stops execution with a non-zero exit code

    return correct_part_idx


def remove_duplicates_with_report(arr):
    arr = np.array(arr)  # Convert the list to a numpy array
    unique_elements, counts = np.unique(arr, return_counts=True)  # Get unique elements and their counts
    duplicates = unique_elements[counts > 1]  # Duplicates are those that appear more than once
    unique_list = unique_elements.tolist()  # Convert unique elements back to a list

    return unique_list, duplicates.tolist()  # Return unique and duplicate lists


def create_gc_part_idx_dict(part, proc_data, it, snapshot):
    it_id = iteration_name(it)
    snap_id = snapshot_name(snapshot)

    gc_id_snap = proc_data[it_id]["snapshots"][snap_id]["gc_id"][()]

    ptype_byte_snap = proc_data[it_id]["snapshots"][snap_id]["ptype"]
    ptype_snap = [ptype.decode("utf-8") for ptype in ptype_byte_snap]

    # Step 1: group GCs by particle type
    gc_by_ptype = {}
    gc_by_ptype["star"] = []
    gc_by_ptype["dark"] = []

    for gc, ptype in zip(gc_id_snap, ptype_snap):
        gc_by_ptype[ptype].append(gc)

    # Step 2: for each ptype, build a small dict: gc_id → index
    id_idx_map = {}

    for ptype, gc_ids in gc_by_ptype.items():
        ids = part[ptype]["id"]  # potentially millions of entries
        # gc_ids = np.array(gc_ids)  # small subset

        # Check which of these are in the main list
        mask = np.isin(ids, gc_ids)
        idxs = np.nonzero(mask)[0]
        found_ids = ids[idxs]

        # Build small, efficient lookup: GC ID → array index
        id_idx_map[ptype] = dict(zip(found_ids, idxs))

        # concerned abour duplciate star ids
        if ptype == "star":
            _, duplicates_ids = remove_duplicates_with_report(found_ids)

    # only concerned with duplciates in star
    for gc_id in id_idx_map["star"].keys():
        if gc_id in duplicates_ids:
            ana_mask = proc_data[it_id]["source"]["analyse_flag"][()] == 1
            gc_mask = proc_data[it_id]["source"]["gc_id"][()] == gc_id
            snapform = proc_data[it_id]["source"]["snap_zform"][ana_mask & gc_mask][0]

            corrected_idx = get_correct_gc_part_idx(part, gc_id, snapform, snapshot)

            id_idx_map["star"][gc_id] = corrected_idx

    return id_idx_map, gc_id_snap, ptype_snap

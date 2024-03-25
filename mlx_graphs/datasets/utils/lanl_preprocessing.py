import gzip
from typing import Callable

from tqdm import tqdm

"""Preprocessing code borrowed from https://github.com/iHeartGraph/Euler"""

DELTA = 1 * 60  # 1 min window per file


def mark_anoms(redteam_file: str) -> dict:
    """
    Parses the redteam file and creates a small dict of
    nodes involved with anomalous edges, and when they occur
    """
    with gzip.open(redteam_file, "rt") as f:
        red_events = f.read().split()

    # Slice out header
    red_events = red_events[1:]

    def add_ts(d, val, ts):
        # val = (src, dst)
        val = (val[1], val[2])
        if val in d:
            d[val].append(ts)
        else:
            d[val] = [ts]

    anom_dict = {}
    for event in red_events:
        tokens = event.split(",")
        ts = int(tokens.pop(0))
        add_ts(anom_dict, tokens, ts)

    return anom_dict


def is_anomalous(d: dict, src: str, dst: str, ts: int) -> bool:
    if ts < 150885 or (src, dst) not in d:
        return False

    times = d[(src, dst)]
    for time in times:
        # Mark true if node appeared in a comprimise
        # in the last 24 hrs (as was done by Nethawk)
        if ts == time:
            return True

    return False


def split(
    auth_file: str,
    redteam_file: str,
    dst_path: str,
    graph_path_at_index_fn: Callable,
    duration_per_file: int = DELTA,
):
    file_cnt = 0
    anom_dict = mark_anoms(redteam_file)

    last_time = 1
    cur_time = 0

    f_in = gzip.open(auth_file, "r")
    f_out = open(dst_path + str(cur_time) + ".csv", "w+")

    line = f_in.readline().decode()  # Skip headers
    line = f_in.readline().decode()

    nmap = {}
    nid = [0]

    usr_nmap = {}
    usr_nid = [0]

    def get_or_add(n):
        if n not in nmap:
            nmap[n] = nid[0]
            nid[0] += 1

        return nmap[n]

    def get_or_add_usr(n):
        if n not in usr_nmap:
            usr_nmap[n] = usr_nid[0]
            usr_nid[0] += 1

        return usr_nmap[n]

    prog = tqdm(desc="Seconds parsed", total=5011199)

    def fmt_label(ts, src, dst):
        return 1 if is_anomalous(anom_dict, src, dst, ts) else 0

    def fmt_usr(x):
        return x.split("@")[0].replace("$", "")

    # Really only care about time stamp, and src/dst computers
    # Hopefully this saves a bit of space when replicating the huge
    # auth.txt flow file
    def fmt_line(ts, src, dst, success_failure, src_user, dst_user):
        return (
            "%s,%s,%s,%s,%s,%s,u%s,u%s\n"
            % (
                ts,
                get_or_add(src),
                get_or_add(dst),
                fmt_label(int(ts), src, dst),
                1 if success_failure == "Success\n" else 0,
                1
                if src_user[0].lower() == "u"
                else (
                    2 if src_user[0].lower() == "c" else 3
                ),  # user, computer or anonymous login as in Argus
                get_or_add_usr(fmt_usr(src_user)),
                get_or_add_usr(fmt_usr(dst_user)),
            ),
            int(ts),
        )

    edges_per_snap = [set()]

    while line:
        # Some filtering for better FPR/less Kerb noise
        if "NTLM" not in line.upper():
            line = f_in.readline().decode()
            continue

        tokens = line.split(",")

        fmt_l, ts = fmt_line(
            tokens[0], tokens[3], tokens[4], tokens[8], tokens[1], tokens[2]
        )

        edges_per_snap[-1].add((get_or_add(tokens[3]), get_or_add(tokens[4])))

        if ts != last_time:
            prog.update(ts - last_time)
            last_time = ts

        if ts >= cur_time + duration_per_file:
            cur_time += duration_per_file
            f_out.close()
            dst_file = graph_path_at_index_fn(file_cnt)
            f_out = open(dst_file, "w+")
            file_cnt += 1
            edges_per_snap.append(set())

        f_out.write(fmt_l)
        line = f_in.readline().decode()

    f_out.close()
    f_in.close()

    print("Auth parsing done.")

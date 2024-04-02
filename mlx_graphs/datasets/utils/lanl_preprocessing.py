import gzip
import os
import pickle

from tqdm import tqdm

"""Preprocessing code borrowed from https://github.com/iHeartGraph/Euler"""

DELTA = 60  # 1 min window per file


def mark_anoms(redteam_file: str):
    with gzip.open(redteam_file, "rt") as f:
        red_events = f.read().split()
    red_events = red_events[1:]

    def add_ts(d, val, ts):
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


def is_anomalous(d, src, dst, ts):
    if ts < 150885 or (src, dst) not in d:
        return False
    times = d[(src, dst)]
    for time in times:
        if ts == time:
            return True
    return False


def is_anomalous_range(d, src, dst, ts):
    if ts < 150885 or (src, dst) not in d:
        return False
    times = d[(src, dst)]
    for time in times:
        if abs(ts - time) <= 300:
            return True
    return False


# comparing ts to time -/+ 5min, only one node is used
def is_anomalous_node_range(d, node, ts):
    if ts < 150885 or node not in d:
        return False

    times = d[node]
    for time in times:
        # Mark true if node appeared in a compromise in -/5min
        if abs(ts - time) <= 300:
            return True

    return False


def save_map(m, fname, dst_path):
    m_rev = [None] * (max(m.values()) + 1)
    for k, v in m.items():
        m_rev[v] = k

    with open(os.path.join(dst_path, fname), "wb") as f:
        pickle.dump(m_rev, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(dst_path + fname + " saved")


def get_or_add(n, m, id):
    if n not in m:
        m[n] = id[0]
        id[0] += 1

    return m[n]


def split(
    auth_file: str,
    redteam_file: str,
    dst_path: str,
    duration_per_file: int = DELTA,
):
    anom_dict = mark_anoms(redteam_file)

    last_time = 1
    cur_time = 0

    f_in = gzip.open(auth_file, "r")
    f_out = open(os.path.join(dst_path, f"graph_{cur_time}.csv"), "w")

    line = f_in.readline().decode()  # Skip headers
    line = f_in.readline().decode()

    nmap = {}
    nid = [0]
    umap = {}
    uid = [0]
    atmap = {}
    atid = [0]
    ltmap = {}
    ltid = [0]
    aomap = {}
    aoid = [0]
    smap = {}
    sid = [0]
    prog = tqdm(desc="Seconds parsed", total=5011199)

    fmt_src = lambda x: x.split("@")[0].replace("$", "")  #  noqa E731

    fmt_label = lambda ts, src, dst: 1 if is_anomalous(anom_dict, src, dst, ts) else 0  #  noqa E731

    fmt_line = lambda ts, src, dst, src_u, dst_u, auth_t, logon_t, auth_o, success: (  #  noqa E731
        "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
        % (
            ts,
            get_or_add(src, nmap, nid),
            get_or_add(dst, nmap, nid),
            get_or_add(fmt_src(src_u), umap, uid),
            get_or_add(fmt_src(dst_u), umap, uid),
            get_or_add(auth_t, atmap, atid),
            get_or_add(logon_t, ltmap, ltid),
            get_or_add(auth_o, aomap, aoid),
            get_or_add(success, smap, sid),
            fmt_label(int(ts), src, dst),
        ),
        int(ts),
    )
    # ts, src, dst, src_user, dst_user, auth_type, logon_type, auth_orientation, success

    while line:
        # Some filtering for better FPR/less Kerb noise
        if "NTLM" not in line.upper():
            line = f_in.readline().decode()
            continue

        tokens = line.split(",")
        # 0: ts, 1: src_u, 2: dest_u, 3: src_c, 4: dest_c, 5:auth_type,
        # 6: logon_type, 7: auth_orientation, 8: success/failure
        # last field has '\n', need to be removed
        line_l, ts = fmt_line(
            tokens[0],
            tokens[3],
            tokens[4],
            tokens[1],
            tokens[2],
            tokens[5],
            tokens[6],
            tokens[7],
            tokens[8][:-1],
        )

        if ts != last_time:
            prog.update(ts - last_time)
            last_time = ts

        # After ts progresses at least 10,000 seconds, make a new file
        if ts >= cur_time + duration_per_file:
            f_out.close()
            cur_time += duration_per_file
            dst_file = os.path.join(
                dst_path, f"graph_{cur_time//duration_per_file}.csv"
            )
            f_out = open(dst_file, "w+")

        f_out.write(line_l)
        line = f_in.readline().decode()

    f_out.close()
    f_in.close()

    save_map(nmap, "nmap.pkl", dst_path)
    save_map(umap, "umap.pkl", dst_path)
    save_map(atmap, "atmap.pkl", dst_path)
    save_map(ltmap, "ltmap.pkl", dst_path)
    save_map(aomap, "aomap.pkl", dst_path)
    save_map(smap, "smap.pkl", dst_path)


def reverse_load_map(auth_path, fname):
    # mapping pickle is a list, need to reverse it to a dict
    m = {}

    with open(os.path.join(auth_path, fname), "rb") as f:
        line_l = pickle.load(f)
        for i in range(0, len(line_l)):
            m[line_l[i]] = i
    return m


def split_flows(
    flows_path: str,
    redteam_file: str,
    dst_path: str,
    auth_path: str,
    duration_per_file: int = DELTA,
):
    anom_dict = mark_anoms(redteam_file)

    last_time = 1
    cur_time = 0

    f_in = gzip.open(flows_path, "r")

    os.makedirs(dst_path, exist_ok=True)

    f_out = open(os.path.join(dst_path, f"flows_{cur_time}.csv"), "w")

    line = f_in.readline().decode()

    nmap = reverse_load_map(auth_path, "nmap.pkl")

    # port mapping
    port_map = {}
    port_id = [0]
    # protocol mapping
    proto_map = {}
    proto_id = [0]

    # the total is read from the last line
    prog = tqdm(desc="Seconds parsed", total=3126928)

    fmt_label = (  #  noqa E731
        lambda ts, src, dst: 1 if is_anomalous_range(anom_dict, src, dst, ts) else 0
    )

    # 0: ts, 1: duration, 2: source computer, 3: source port, 4: destination computer,
    # 5: destination port, 6: protocol, 7: packet count, 8: byte count
    fmt_line = lambda ts, duration, src, src_p, dst, dst_p, proto, pkt_cnt, byte_cnt: (  #  noqa E731
        "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
        % (
            ts,
            nmap[src],
            nmap[dst],
            get_or_add(src_p, port_map, port_id),
            get_or_add(dst_p, port_map, port_id),
            get_or_add(proto, proto_map, proto_id),
            duration,
            pkt_cnt,
            byte_cnt,
            fmt_label(int(ts), src, dst),
        ),
        int(ts),
    )

    while line:
        if "?" in line:
            line = f_in.readline().decode()
            continue

        tokens = line.split(",")

        # Filtering out the lines with src and dest that are not in auth
        if tokens[2] not in nmap or tokens[4] not in nmap:
            line = f_in.readline().decode()
            continue

        # last field has '\n', need to be removed
        line_l, ts = fmt_line(
            tokens[0],
            tokens[1],
            tokens[2],
            tokens[3],
            tokens[4],
            tokens[5],
            tokens[6],
            tokens[7],
            tokens[8][:-1],
        )

        if ts != last_time:
            prog.update(ts - last_time)
            last_time = ts

        if ts >= cur_time + duration_per_file:
            f_out.close()
            cur_time += duration_per_file
            f_out = open(os.path.join(dst_path, f"flows_{cur_time}.csv"), "w")

        f_out.write(line_l)
        line = f_in.readline().decode()

    f_out.close()
    f_in.close()

    save_map(port_map, "pomap.pkl", dst_path)
    save_map(proto_map, "prmap.pkl", dst_path)


if __name__ == "__main__":
    split()
    split_flows()

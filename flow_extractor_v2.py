"""
Bidirectional Flow Extractor v2
===============================
Converts pcap files to CSV flow records with proper timeout-based
flow aggregation and enriched feature set.

Changes from v1:
  - Idle timeout (60s) and active timeout (120s) to split long-lived
    connections into multiple flow records, matching NetFlow/IPFIX behaviour.
  - Expanded from 13 to 22 numeric features for richer ML signal.
  - Added per-direction byte/packet stats, ratios, and flag counts.

Usage:
    python flow_extractor_v2.py <input.pcap> <output.csv> <label>
"""

from scapy.all import PcapReader, IP, IPv6, TCP, UDP, ICMP
import csv
import sys
from math import inf
from collections import defaultdict


# ── Timeouts (seconds) ──────────────────────────────────────────
IDLE_TIMEOUT = 60       # No packet for 60s → flow is finished
ACTIVE_TIMEOUT = 120    # Flow open for 120s → export and start new
# ────────────────────────────────────────────────────────────────


def get_flow_key(pkt):
    """Return (canonical_key, direction) or None if not IP."""
    ip_layer = pkt.getlayer(IP) or pkt.getlayer(IPv6)
    if ip_layer is None:
        return None

    proto = None
    sport = dport = 0

    if pkt.haslayer(TCP):
        l = pkt.getlayer(TCP)
        proto = "TCP"
        sport, dport = l.sport, l.dport
    elif pkt.haslayer(UDP):
        l = pkt.getlayer(UDP)
        proto = "UDP"
        sport, dport = l.sport, l.dport
    elif pkt.haslayer(ICMP):
        proto = "ICMP"
    else:
        try:
            proto = str(ip_layer.proto)
        except AttributeError:
            return None

    src = ip_layer.src
    dst = ip_layer.dst

    fwd = (src, sport, dst, dport, proto)
    bwd = (dst, dport, src, sport, proto)
    if fwd <= bwd:
        return fwd, "fwd"
    else:
        return bwd, "bwd"


def new_flow(ts):
    """Initialise a fresh flow record."""
    return {
        "first_ts": ts,
        "last_ts": ts,
        # Packet counts
        "pkt_count_fwd": 0,
        "pkt_count_bwd": 0,
        # Byte counts
        "byte_count_fwd": 0,
        "byte_count_bwd": 0,
        # Packet sizes for std dev
        "pkt_sizes_fwd": [],
        "pkt_sizes_bwd": [],
        # Inter-arrival times
        "last_ts_prev": ts,
        "iat_list": [],
        # TCP flags (fwd direction)
        "syn_count": 0,
        "ack_count": 0,
        "fin_count": 0,
        "rst_count": 0,
        "psh_count": 0,
    }


def export_flow(key, f, label):
    """Convert internal flow dict to an output row dict."""
    src, sport, dst, dport, proto = key

    dur = max(f["last_ts"] - f["first_ts"], 0.0)

    pkt_fwd = f["pkt_count_fwd"]
    pkt_bwd = f["pkt_count_bwd"]
    pkt_total = pkt_fwd + pkt_bwd

    byte_fwd = f["byte_count_fwd"]
    byte_bwd = f["byte_count_bwd"]
    byte_total = byte_fwd + byte_bwd

    # IAT statistics
    iats = f["iat_list"]
    if len(iats) > 0:
        mean_iat = sum(iats) / len(iats)
        min_iat = min(iats)
        max_iat = max(iats)
        # Standard deviation of IAT
        if len(iats) > 1:
            variance = sum((x - mean_iat) ** 2 for x in iats) / (len(iats) - 1)
            std_iat = variance ** 0.5
        else:
            std_iat = 0.0
    else:
        mean_iat = min_iat = max_iat = std_iat = 0.0

    # Packet size statistics
    all_sizes = f["pkt_sizes_fwd"] + f["pkt_sizes_bwd"]
    if len(all_sizes) > 0:
        mean_pkt_size = sum(all_sizes) / len(all_sizes)
        min_pkt_size = min(all_sizes)
        max_pkt_size = max(all_sizes)
    else:
        mean_pkt_size = min_pkt_size = max_pkt_size = 0.0

    # Derived ratio features
    bytes_per_pkt = byte_total / pkt_total if pkt_total > 0 else 0.0
    pkt_rate = pkt_total / dur if dur > 0 else 0.0
    byte_rate = byte_total / dur if dur > 0 else 0.0
    fwd_bwd_pkt_ratio = pkt_fwd / (pkt_bwd + 1)
    fwd_bwd_byte_ratio = byte_fwd / (byte_bwd + 1)

    return {
        "src_ip": src,
        "src_port": sport,
        "dst_ip": dst,
        "dst_port": dport,
        "protocol": proto,
        "flow_duration": round(dur, 6),
        # Packet counts
        "pkt_count_fwd": pkt_fwd,
        "pkt_count_bwd": pkt_bwd,
        "pkt_count_total": pkt_total,
        # Byte counts
        "byte_count_fwd": byte_fwd,
        "byte_count_bwd": byte_bwd,
        "byte_count_total": byte_total,
        # IAT features
        "mean_iat": round(mean_iat, 6),
        "min_iat": round(min_iat, 6),
        "max_iat": round(max_iat, 6),
        "std_iat": round(std_iat, 6),
        # Packet size features
        "mean_pkt_size": round(mean_pkt_size, 2),
        "min_pkt_size": min_pkt_size,
        "max_pkt_size": max_pkt_size,
        # Rate features
        "bytes_per_pkt": round(bytes_per_pkt, 2),
        "pkt_rate": round(pkt_rate, 2),
        "byte_rate": round(byte_rate, 2),
        # Ratio features
        "fwd_bwd_pkt_ratio": round(fwd_bwd_pkt_ratio, 4),
        "fwd_bwd_byte_ratio": round(fwd_bwd_byte_ratio, 4),
        # TCP flag counts
        "syn_count": f["syn_count"],
        "ack_count": f["ack_count"],
        "fin_count": f["fin_count"],
        "rst_count": f["rst_count"],
        "psh_count": f["psh_count"],
        # Label
        "label": label,
    }


FIELDNAMES = [
    "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
    "flow_duration",
    "pkt_count_fwd", "pkt_count_bwd", "pkt_count_total",
    "byte_count_fwd", "byte_count_bwd", "byte_count_total",
    "mean_iat", "min_iat", "max_iat", "std_iat",
    "mean_pkt_size", "min_pkt_size", "max_pkt_size",
    "bytes_per_pkt", "pkt_rate", "byte_rate",
    "fwd_bwd_pkt_ratio", "fwd_bwd_byte_ratio",
    "syn_count", "ack_count", "fin_count", "rst_count", "psh_count",
    "label",
]


def extract_flows(pcap_path, out_csv, label):
    """Read pcap, aggregate packets into timeout-based flows, write CSV."""
    active_flows = {}
    exported_rows = []

    def export_and_remove(key):
        """Export a flow and remove it from active tracking."""
        row = export_flow(key, active_flows[key], label)
        exported_rows.append(row)
        del active_flows[key]

    pkt_count = 0

    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            if not (pkt.haslayer(IP) or pkt.haslayer(IPv6)):
                continue

            try:
                ts = float(pkt.time)
            except Exception:
                continue

            res = get_flow_key(pkt)
            if res is None:
                continue
            key, direction = res

            pkt_count += 1
            if pkt_count % 500000 == 0:
                print(f"  Processed {pkt_count:,} packets, "
                      f"{len(active_flows):,} active flows, "
                      f"{len(exported_rows):,} exported...")

            # ── Check timeouts for this flow ────────────────────
            if key in active_flows:
                f = active_flows[key]
                idle_gap = ts - f["last_ts"]
                active_dur = ts - f["first_ts"]

                if idle_gap > IDLE_TIMEOUT or active_dur > ACTIVE_TIMEOUT:
                    # Export the old flow, start a new one
                    export_and_remove(key)

            # ── Create new flow if needed ───────────────────────
            if key not in active_flows:
                active_flows[key] = new_flow(ts)

            f = active_flows[key]

            # ── Update flow stats ───────────────────────────────
            length = len(pkt)

            if direction == "fwd":
                f["pkt_count_fwd"] += 1
                f["byte_count_fwd"] += length
                f["pkt_sizes_fwd"].append(length)
            else:
                f["pkt_count_bwd"] += 1
                f["byte_count_bwd"] += length
                f["pkt_sizes_bwd"].append(length)

            # Inter-arrival time
            iat = ts - f["last_ts_prev"]
            if iat > 0:
                f["iat_list"].append(iat)
            f["last_ts_prev"] = ts
            f["last_ts"] = ts

            # TCP flags
            if pkt.haslayer(TCP):
                flags = pkt.getlayer(TCP).flags
                if flags & 0x02:  # SYN
                    f["syn_count"] += 1
                if flags & 0x10:  # ACK
                    f["ack_count"] += 1
                if flags & 0x01:  # FIN
                    f["fin_count"] += 1
                if flags & 0x04:  # RST
                    f["rst_count"] += 1
                if flags & 0x08:  # PSH
                    f["psh_count"] += 1

    # ── Export remaining active flows ───────────────────────────
    for key in list(active_flows.keys()):
        export_and_remove(key)

    # ── Write CSV ───────────────────────────────────────────────
    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in exported_rows:
            writer.writerow(row)

    print(f"\nDone: {pkt_count:,} packets → {len(exported_rows):,} flows")
    print(f"Saved to: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python flow_extractor_v2.py <input.pcap> <output.csv> <label>")
        sys.exit(1)
    extract_flows(sys.argv[1], sys.argv[2], sys.argv[3])

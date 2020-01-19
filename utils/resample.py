#!/usr/bin/env python

"""
Resample font
"""

import argparse
import math
import numpy as np
from numpy.linalg import norm
import heapq
import json

def parse_args():
    parser = argparse.ArgumentParser(__doc__);
    parser.add_argument("--num-samples", "-N", type=int, default=100,
            help="Number of samples");
    parser.add_argument("font_file");
    parser.add_argument("character");
    parser.add_argument("output_file");
    return parser.parse_args();

def unpack_font_file(font_file):
    with open(font_file, 'rb') as fin:
        font_data = json.load(fin);
        return font_data;

def parse_font_file(font_data):
    lines = font_data.split("\n");
    num_lines = len(lines);
    line_number = 0;
    while line_number < num_lines:
        line = lines[line_number];
        line_number+=1;
        num_control_pts = int(line.strip());
        ctrl_pts = np.ndarray((num_control_pts, 2));
        for i in range(num_control_pts):
            line = lines[line_number].strip();
            line_number += 1;
            fields = line.split();
            assert(len(fields)==2);
            ctrl_pts[i,0] = float(fields[0]);
            ctrl_pts[i,1] = float(fields[1]);

        line = lines[line_number].strip();
        line_number += 1;
        num_curves = int(line);
        curves = [];
        for i in range(num_curves):
            line = lines[line_number].strip();
            line_number += 1;
            fields = line.split();
            curves.append([int(val) for val in fields[1:]]);

        return {"ctrl_pts": ctrl_pts,
                "indices": curves};

def chain_curves(data):
    ctrl_pts = data["ctrl_pts"];
    indices = data["indices"];
    num_ctrl_pts = len(ctrl_pts);
    num_curves = len(indices);

    start_to_curve = np.zeros(num_ctrl_pts, dtype=int) + num_curves;
    for i in range(num_curves):
        if (start_to_curve[indices[i][0]] != num_curves):
            print("Multiple curve starts from the same vertex");
            print(i, indices[i][0]);
        start_to_curve[indices[i][0]] = i;

    visited = np.zeros(num_curves);
    chains = [];
    for i in range(num_curves):
        if visited[i] != 0: continue;
        visited[i] = 1;
        chain = [i];

        curve_idx = i;
        next_start_idx = indices[i][-1];
        next_curve = start_to_curve[next_start_idx];
        while next_curve != i:
            if (next_curve >= num_curves):
                # Input contains open curve!
                break;
            visited[next_curve] = 1;
            chain.append(next_curve);
            next_start_idx = indices[next_curve][-1];
            next_curve = start_to_curve[next_start_idx];
        chains.append(chain);

    data["loops"] = chains;

def approximate_arc_length(data, loop):
    ctrl_pts = data["ctrl_pts"];
    segment_lengths = [];
    for curve_id in loop:
        l = 0.0;
        curve = data["indices"][curve_id];
        if len(curve) == 2:
            l += norm(ctrl_pts[curve[0]] - ctrl_pts[curve[1]]);
        elif len(curve) == 3:
            l += norm(ctrl_pts[curve[0]] - ctrl_pts[curve[1]]);
            l += norm(ctrl_pts[curve[1]] - ctrl_pts[curve[2]]);
        elif len(curve) == 4:
            l += norm(ctrl_pts[curve[0]] - ctrl_pts[curve[1]]);
            l += norm(ctrl_pts[curve[1]] - ctrl_pts[curve[2]]);
            l += norm(ctrl_pts[curve[2]] - ctrl_pts[curve[3]]);
        else:
            raise NotImplementedError("Beizer with degree > 3 is not supported");
        segment_lengths.append(l);
    return segment_lengths;

def eval_bezier(pts, t):
    num_pts = len(pts);
    if num_pts == 1:
        return pts[0];
    pts = [pts[i]*(1.0-t) + pts[i+1]*t for i in range(num_pts-1)];
    return eval_bezier(pts, t);

def resample_curve(data, curve_idx, s):
    ctrl_pts = data["ctrl_pts"];
    indices = data["indices"][curve_idx];
    if s == 0:
        return [ctrl_pts[indices[-1]]];
    else:
        ts = np.linspace(0.0, 1.0, num=s+2)[1:];
        if len(indices) == 2:
            p0 = ctrl_pts[indices[0]];
            p1 = ctrl_pts[indices[1]];
            return [p0*(1-t) + p1*t for t in ts ];
        elif len(indices) == 3:
            p0 = ctrl_pts[indices[0]];
            p1 = ctrl_pts[indices[1]];
            p2 = ctrl_pts[indices[2]];
            return [eval_bezier([p0, p1, p2], t) for t in ts ];
        elif len(indices) == 4:
            p0 = ctrl_pts[indices[0]];
            p1 = ctrl_pts[indices[1]];
            p2 = ctrl_pts[indices[2]];
            p3 = ctrl_pts[indices[3]];
            return [eval_bezier([p0, p1, p2, p3], t) for t in ts ];
        else:
            raise NotImplementedError("Higher order bezier is not supported");

def resample(data, N):
    num_ctrl_pts = len(data["ctrl_pts"])
    sampled_loops = [];
    for loop in data["loops"]:
        m = len(loop);
        if (m > N):
            raise RuntimeError("This letter requires at least " + str(m) + " samples");

        segment_lengths = approximate_arc_length(data, loop);
        assert(len(segment_lengths) == m);
        total_length = np.sum(segment_lengths);
        sample_length = total_length / N;

        Q = [];
        for i,l in enumerate(segment_lengths):
            heapq.heappush(Q, (-l, 0, i));

        for i in range(N-m):
            entry = heapq.heappop(Q);
            entry = (entry[0]*(entry[1]+1)/(entry[1]+2), entry[1]+1, entry[2])
            heapq.heappush(Q, entry);

        num_extra_samples = np.zeros(m, dtype=int);
        for entry in Q:
            num_extra_samples[entry[2]] = entry[1];
        assert(np.sum(num_extra_samples) + m == N);

        samples = [];
        for i in range(m):
            curve_id = loop[i];
            s = num_extra_samples[i];
            samples += resample_curve(data, curve_id, s);
        sampled_loops.append(np.vstack(samples));
    return sampled_loops;

def save_loops(output_file, loops):
    with open(output_file, 'w') as fout:
        count = 0;
        for loop in loops:
            n = len(loop);
            for i in range(n):
                fout.write("v {} {} 0\n".format(loop[i,0], loop[i,1]));
            for i in range(n):
                fout.write("l {} {}\n".format(count+i+1, count+(i+1)%n+1));

            count += n;

def main():
    args = parse_args();
    font_data = unpack_font_file(args.font_file);
    if args.character not in font_data:
        raise RuntimeError("Requested character not found in font file!");
    data = parse_font_file(font_data[args.character]);
    chain_curves(data);

    loops = resample(data, args.num_samples);
    save_loops(args.output_file, loops);

if __name__ == "__main__":
    main();

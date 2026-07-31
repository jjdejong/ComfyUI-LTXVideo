#!/usr/bin/env python3
"""Derive a Director-driven looping workflow from the two-pass looping workflow.

Reads the hand-maintained two-pass AV I2V looping graph and produces a variant whose
per-tile prompts and keyframes come from a WhatDreamsCost **LTX Director** timeline via
the **LTX Looping Bridge** node, instead of the MultiPromptProvider + snippet/concatenate
machinery and the LTXVLoopingReferenceSchedule image scheduling.

What it changes (everything leans on the existing KJNodes Set/Get bus fabric, so only a
few SetNode sources are repointed and all consumers follow):

  * Adds LTXDirector (copied verbatim from a known-good serialized node so its
    JS-managed widget layout is correct) + LTXLoopingBridge.
  * Repoints the `tile_prompt_conditioning` bus  <- Bridge.per_tile_conditionings
            the `scheduled_reference_images` bus <- Bridge.cond_images
            the `reference_indices` bus          <- Bridge.cond_image_indices
    (both LTXVLoopingSampler stages read those buses, so they pick up the new sources.)
  * Disconnects LTXVLoopingReferenceSchedule's reference_images input so missing late-ref
    PNGs can't break execution; the node is kept purely for timing (frame_count /
    temporal_tile_size / temporal_overlap buses).
  * Removes the now-obsolete per-tile prompt + late-reference subgraph.

Deliberately KEPT (robustness): the existing bus-driven EmptyLTXVLatentVideo /
LTXVEmptyLatentAudio shells (correct shape from frame_count) and the whole
guider/sampler/upscale/decode/output chain. Director's own video_latent/audio_latent
outputs are left available but unwired — see the README note for switching to
Director-driven custom audio.

Run:  python3 transform_to_director_looping.py
Out:  LTX-2.3_Director_Looping.json
"""

import json
import os

HERE = os.path.dirname(__file__)
BASE = os.path.join(HERE, "LTX-2.3_Two_Pass_I2V_Looping.json")
DIRECTOR_SRC = os.path.join(
    HERE, "..", "..", "WhatDreamsCost-ComfyUI", "example_workflows",
    "LTX_Director_2_Workflow_Hotfix.json",
)

# Director's own output count, i.e. the first slot index our appended passthroughs
# get. Upstream v2.0.4 inserted motion_guide_data at slot 5, shifting frame_rate ->
# 6 and combined_audio -> 7; a workflow generated against the older layout wires
# frame_rate to a MOTION_GUIDE_DATA socket and fails validation. Derived from the
# donor node below rather than hardcoded, so the next upstream insertion is caught.
DIRECTOR_BASE_OUTPUTS = 8
DIRECTOR_SLOT_FRAME_RATE = 6
OUT = os.path.join(HERE, "LTX-2.3_Director_Looping.json")

DIRECTOR_ID = 300
BRIDGE_ID = 301

# Obsolete nodes removed by the transform (rewired first, where needed, below).
REMOVE_CONTENT = [
    81,                      # MultiPromptProvider
    82, 85, 88, 91,          # Late Ref LoadImage
    83, 86, 89, 92,          # Tile snippet PrimitiveStringMultiline
    94, 95, 96, 97,          # ImageBatch (Ref Batch)
    98, 99, 100,             # StringConcatenate (Join Tile Prompts)
    184, 185, 186, 187,      # MergeString
    178, 180,                # SetNode reference_image_batch / joined_tile_prompts
    80,                      # standalone "Global Positive Prompt" primitive (Director owns it)
    # --- consolidation: timing into the bridge ---
    24,                      # LTXVLoopingReferenceSchedule (bridge now emits tile/overlap/frame_count)
    4,                       # fps PrimitiveFloat (fps now comes from Director.frame_rate)
    # --- consolidation: drop positive fallback encode ---
    20,                      # CLIPTextEncode "Global Prompt Fallback Encode" (use Director.positive)
    # --- consolidation: slim first-ref image (frame-0 I2V via Director keyframe) ---
    32,                      # Stage 1 I2V Cond (LTXVImgToVideoConditionOnly)
    52,                      # Stage 2 I2V Cond
    1,                       # LoadImage (reference now taken from bridge.start_image = cond_images[0])
    110,                     # SetNode start_image (dead bus; source was LoadImage)
]


def main():
    with open(BASE) as f:
        wf = json.load(f)
    with open(DIRECTOR_SRC) as f:
        dsrc = json.load(f)

    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}
    next_link = [wf["last_link_id"]]

    def new_link(from_id, from_slot, to_id, to_slot, typ):
        next_link[0] += 1
        lid = next_link[0]
        wf["links"].append([lid, from_id, from_slot, to_id, to_slot, typ])
        links[lid] = wf["links"][-1]
        # bookkeeping
        fo = nodes[from_id]["outputs"][from_slot]
        fo.setdefault("links", [])
        if fo["links"] is None:
            fo["links"] = []
        fo["links"].append(lid)
        nodes[to_id]["inputs"][to_slot]["link"] = lid
        return lid

    def find_set(name):
        for n in wf["nodes"]:
            if n["type"] == "SetNode" and n["widgets_values"][0] == name:
                return n
        raise KeyError(name)

    def rewire_input(to_id, to_slot, from_id, from_slot, typ):
        """Detach whatever currently feeds (to_id, to_slot) and wire it from a new source."""
        ip = nodes[to_id]["inputs"][to_slot]
        old = links.get(ip["link"])
        if old is not None:
            for op in nodes.get(old[1], {}).get("outputs", []) or []:
                if op.get("links"):
                    op["links"] = [x for x in op["links"] if x != old[0]]
            wf["links"] = [x for x in wf["links"] if x[0] != old[0]]
            links.pop(old[0], None)
        ip["link"] = None
        new_link(from_id, from_slot, to_id, to_slot, typ)

    # ── 1. Add LTXDirector (verbatim serialized node, re-id + reposition + rewire) ──
    director = next(n for n in dsrc["nodes"] if n["type"] == "LTXDirector")
    director = json.loads(json.dumps(director))  # deep copy
    director["id"] = DIRECTOR_ID
    director["pos"] = [1150, 1650]
    director["order"] = DIRECTOR_ID
    # Drop inbound links from the source graph; we rewire fresh.
    for ip in director["inputs"]:
        ip["link"] = None
    for op in director.get("outputs", []):
        op["links"] = []
    # Patched LTXDirector exposes three extra passthrough outputs so the bridge can be
    # wired directly instead of pasting the widget values. They are appended after
    # Director's own outputs, so their slot indices follow the donor's output count --
    # never hardcode them (see DIRECTOR_BASE_OUTPUTS).
    base = len(director["outputs"])
    if base != DIRECTOR_BASE_OUTPUTS:
        raise SystemExit(
            f"Donor LTXDirector has {base} outputs, expected {DIRECTOR_BASE_OUTPUTS}. "
            "Upstream changed the output layout; update DIRECTOR_BASE_OUTPUTS and "
            "DIRECTOR_SLOT_FRAME_RATE, then re-check every Director slot reference below."
        )
    for offset, name in enumerate(("local_prompts", "segment_lengths", "global_prompt")):
        director["outputs"].append(
            {"name": name, "type": "STRING", "links": [], "slot_index": base + offset}
        )
    SLOT_LOCAL_PROMPTS = base
    SLOT_SEGMENT_LENGTHS = base + 1
    SLOT_GLOBAL_PROMPT = base + 2
    wf["nodes"].append(director)
    nodes[DIRECTOR_ID] = director

    # ── 2. Add LTXLoopingBridge (2 input slots + 6 widgets) ──
    global_prompt = nodes[80]["widgets_values"][0]
    bridge = {
        "id": BRIDGE_ID,
        "type": "LTXLoopingBridge",
        "pos": [1700, 1650],
        "size": [400, 220],
        "flags": {},
        "order": BRIDGE_ID,
        "mode": 0,
        "inputs": [
            {"name": "clip", "type": "CLIP", "link": None},
            {"name": "guide_data", "type": "GUIDE_DATA", "link": None},
            # Widget-converted inputs (value kept in widgets_values as fallback; the link wins).
            # temporal_tile_size / temporal_overlap stay pure WIDGETS (set here, emitted as outputs).
            {"name": "local_prompts", "type": "STRING", "link": None, "widget": {"name": "local_prompts"}},
            {"name": "segment_lengths", "type": "STRING", "link": None, "widget": {"name": "segment_lengths"}},
            {"name": "global_prompt", "type": "STRING", "link": None, "widget": {"name": "global_prompt"}},
            {"name": "frame_rate", "type": "FLOAT", "link": None, "widget": {"name": "frame_rate"}},
        ],
        "outputs": [
            {"name": "per_tile_conditionings", "type": "CONDITIONING", "links": [], "slot_index": 0},
            {"name": "cond_images", "type": "IMAGE", "links": [], "slot_index": 1},
            {"name": "cond_image_indices", "type": "STRING", "links": [], "slot_index": 2},
            {"name": "temporal_tile_size", "type": "INT", "links": [], "slot_index": 3},
            {"name": "temporal_overlap", "type": "INT", "links": [], "slot_index": 4},
            {"name": "frame_count", "type": "INT", "links": [], "slot_index": 5},
            {"name": "start_image", "type": "IMAGE", "links": [], "slot_index": 6},
        ],
        "properties": {"Node name for S&R": "LTXLoopingBridge"},
        # widgets_values is positional over every widget-eligible input, ordered
        # REQUIRED FIRST, THEN OPTIONAL -- not by declaration order. temporal_tile_size
        # and temporal_overlap are declared without optional=True, so they land in
        # required and sort ahead of the optional strings, even though they appear
        # later in the source. Using declaration order here fed them the leading ""
        # entries and they failed INT conversion. Verify against
        #   curl -s localhost:8188/object_info/LTXLoopingBridge
        # clip and guide_data are link-only types and take no widget slot.
        # order: temporal_tile_size, temporal_overlap, local_prompts,
        #        segment_lengths, timeline_data, global_prompt, frame_rate
        "widgets_values": [240, 64, "", "", "", global_prompt, 24],
        "title": "LTX Looping Bridge",
    }
    wf["nodes"].append(bridge)
    nodes[BRIDGE_ID] = bridge

    # ── 3. Wire Director + Bridge ──
    new_link(25, 0, DIRECTOR_ID, 0, "MODEL")   # plain model -> Director.model
    new_link(11, 0, DIRECTOR_ID, 1, "CLIP")    # LTXAV CLIP   -> Director.clip
    new_link(12, 0, DIRECTOR_ID, 2, "VAE")     # audio VAE    -> Director.audio_vae
    new_link(11, 0, BRIDGE_ID, 0, "CLIP")      # CLIP         -> Bridge.clip
    new_link(DIRECTOR_ID, 4, BRIDGE_ID, 1, "GUIDE_DATA")     # guide_data      -> Bridge
    new_link(DIRECTOR_ID, SLOT_LOCAL_PROMPTS, BRIDGE_ID, 2, "STRING")     # local_prompts   -> Bridge (auto, no paste)
    new_link(DIRECTOR_ID, SLOT_SEGMENT_LENGTHS, BRIDGE_ID, 3, "STRING")   # segment_lengths -> Bridge (auto, no paste)

    get_id = [310]

    def add_get(bus_name, typ, pos, to_id, to_slot):
        nid = get_id[0]
        get_id[0] += 1
        gn = {
            "id": nid, "type": "GetNode", "pos": list(pos), "size": [190, 58],
            "flags": {"collapsed": True}, "order": nid, "mode": 0,
            "inputs": [],
            "outputs": [{"name": typ, "type": typ, "links": [], "slot_index": 0}],
            "title": f"Get_{bus_name}",
            "properties": {"Node name for S&R": "GetNode", "aux_id": "kijai/ComfyUI-KJNodes"},
            "widgets_values": [bus_name],
        }
        wf["nodes"].append(gn)
        nodes[nid] = gn
        new_link(nid, 0, to_id, to_slot, typ)

    # Bridge global_prompt + frame_rate come from the buses (now sourced from Director, below).
    # temporal_tile_size / temporal_overlap are now the bridge's own widgets (not wired in).
    add_get("global_prompt", "STRING", (1640, 1760), BRIDGE_ID, 4)
    add_get("fps", "FLOAT", (1640, 1820), BRIDGE_ID, 5)

    # ── 4. Repoint SetNode sources to Bridge / Director outputs ──
    def repoint(set_name, new_from_id, new_from_slot):
        sn = find_set(set_name)
        lid = sn["inputs"][0]["link"]
        l = links[lid]
        old_from = l[1]
        # detach from old source's outputs bookkeeping
        for op in nodes[old_from].get("outputs", []):
            if op.get("links"):
                op["links"] = [x for x in op["links"] if x != lid]
        l[1] = new_from_id
        l[2] = new_from_slot
        nodes[new_from_id]["outputs"][new_from_slot]["links"].append(lid)

    repoint("tile_prompt_conditioning", BRIDGE_ID, 0)
    repoint("scheduled_reference_images", BRIDGE_ID, 1)
    repoint("reference_indices", BRIDGE_ID, 2)
    # Director's global_prompt passthrough is the single global source.
    repoint("global_prompt", DIRECTOR_ID, SLOT_GLOBAL_PROMPT)
    # Timing single-sourced from the bridge; fps from Director.frame_rate.
    repoint("temporal_tile_size", BRIDGE_ID, 3)
    repoint("temporal_overlap", BRIDGE_ID, 4)
    repoint("frame_count", BRIDGE_ID, 5)
    repoint("fps", DIRECTOR_ID, DIRECTOR_SLOT_FRAME_RATE)

    # ── 5. Rewire consumers of removed nodes ──
    # Guider base positive from Director.positive (slot 1) instead of the dropped encode (20).
    rewire_input(22, 0, DIRECTOR_ID, 1, "CONDITIONING")
    # Drop the I2V cond nodes (32/52): feed the empty/upscaled video latents straight to the
    # AV concats; frame-0 conditioning now comes from a Director frame-0 keyframe via the bridge.
    rewire_input(33, 0, 30, 0, "LATENT")   # Stage 1 AV concat video <- empty video latent
    rewire_input(53, 0, 51, 0, "LATENT")   # Stage 2 AV concat video <- spatial upscaler
    # Single image source: take the start frame from the bridge's first keyframe (cond_images[0])
    # for output dimensions + the identity anchor, so the separate LoadImage is removed.
    rewire_input(16, 0, BRIDGE_ID, 6, "IMAGE")   # GetImageSize  <- bridge.start_image
    rewire_input(2, 0, BRIDGE_ID, 6, "IMAGE")    # LTXVPreprocess <- bridge.start_image

    # ── 6. Remove obsolete subgraph + dangling links ──
    remove = set(REMOVE_CONTENT)
    wf["links"] = [
        l for l in wf["links"] if l[1] not in remove and l[3] not in remove
    ]
    live = {l[0] for l in wf["links"]}
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in remove]
    nodes = {n["id"]: n for n in wf["nodes"]}
    # clean bookkeeping for surviving nodes
    for n in wf["nodes"]:
        for ip in n.get("inputs", []):
            if ip.get("link") not in live:
                ip["link"] = None
        for op in n.get("outputs", []):
            if op.get("links"):
                op["links"] = [x for x in op["links"] if x in live]

    # ── 7. Prune GetNodes that now feed nothing ──
    def get_consumed(nid):
        return any(l[1] == nid for l in wf["links"])

    pruned = [
        n["id"] for n in wf["nodes"]
        if n["type"] == "GetNode" and not get_consumed(n["id"])
    ]
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in set(pruned)]
    nodes = {n["id"]: n for n in wf["nodes"]}

    wf["last_node_id"] = max(n["id"] for n in wf["nodes"])
    wf["last_link_id"] = next_link[0]
    wf.setdefault("extra", {}).setdefault("info", {})
    wf["extra"]["info"] = {
        "name": "LTX-2.3 Director Looping",
        "description": (
            "Two-pass AV I2V looping driven by an LTX Director timeline through the "
            "LTX Looping Bridge: one prompt per temporal tile (by timeline position) + "
            "keyframes. Author the Director timeline, then paste its local_prompts and "
            "segment_lengths into the bridge."
        ),
    }

    # VHS_VideoCombine serializes its last preview into widgets_values, which bakes an
    # absolute output path from whoever last ran the base workflow. It is stale UI state,
    # not config -- VHS rebuilds it on the next run -- so drop it rather than ship it.
    stripped = 0
    for n in wf["nodes"]:
        wv = n.get("widgets_values")
        if isinstance(wv, dict) and wv.pop("videopreview", None) is not None:
            stripped += 1

    validate(wf)

    with open(OUT, "w") as f:
        json.dump(wf, f, indent=2)
    print(f"Wrote {OUT}")
    if stripped:
        print(f"  stripped stale videopreview from {stripped} node(s)")
    print(f"  {len(wf['nodes'])} nodes, {len(wf['links'])} links; pruned GetNodes: {pruned}")


def validate(wf):
    nodes = {n["id"]: n for n in wf["nodes"]}
    link_ids = set()
    for l in wf["links"]:
        lid, fid, fs, tid, ts, _ = l
        link_ids.add(lid)
        assert fid in nodes, f"link {lid}: missing from-node {fid}"
        assert tid in nodes, f"link {lid}: missing to-node {tid}"
        assert fs < len(nodes[fid].get("outputs", [])), f"link {lid}: bad from-slot {fs} on {fid}"
        assert ts < len(nodes[tid].get("inputs", [])), f"link {lid}: bad to-slot {ts} on {tid}"
    for n in wf["nodes"]:
        for i, ip in enumerate(n.get("inputs", [])):
            lk = ip.get("link")
            assert lk is None or lk in link_ids, f"node {n['id']} input[{i}] dangling link {lk}"
        for op in n.get("outputs", []):
            for lk in (op.get("links") or []):
                assert lk in link_ids, f"node {n['id']} output dangling link {lk}"
    print("  validate: OK (all links reference live nodes/slots)")


if __name__ == "__main__":
    main()

"""Quick GLB inspector — dump mesh/material/node/morph info to stdout.

Usage:  python scripts/inspect_glb.py <path-to-glb>
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: inspect_glb.py <path>")
        return
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"missing: {path}")
        return

    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"glTF":
            print(f"not a GLB: magic={magic!r}")
            return
        version = struct.unpack("<I", f.read(4))[0]
        length = struct.unpack("<I", f.read(4))[0]
        chunk_len = struct.unpack("<I", f.read(4))[0]
        chunk_type = f.read(4)
        if chunk_type != b"JSON":
            print(f"first chunk is not JSON: {chunk_type!r}")
            return
        data = json.loads(f.read(chunk_len))

    print(f"=== {path.name}  glTF v{version}  total={length:,} bytes ===")
    print(f"JSON chunk size: {chunk_len:,} bytes")
    print()

    meshes = data.get("meshes", [])
    print(f"=== Meshes ({len(meshes)}) ===")
    for i, m in enumerate(meshes):
        name = m.get("name", "?")
        prims = m.get("primitives", [])
        morph_count = sum(len(p.get("targets", [])) for p in prims)
        print(f"  [{i}] {name}  primitives={len(prims)}  morph_targets={morph_count}")
        target_names = m.get("extras", {}).get("targetNames")
        if target_names:
            for j, t in enumerate(target_names):
                print(f"        [{j:>3}] {t}")
    print()

    materials = data.get("materials", [])
    print(f"=== Materials ({len(materials)}) ===")
    for i, mat in enumerate(materials):
        print(f"  [{i}] {mat.get('name', '?')}")
    print()

    nodes = data.get("nodes", [])
    mesh_nodes = [
        (i, n.get("name"), n.get("mesh"))
        for i, n in enumerate(nodes)
        if n.get("mesh") is not None
    ]
    print(f"=== Nodes total={len(nodes)}, mesh-bearing={len(mesh_nodes)} ===")
    for idx, name, mesh_idx in mesh_nodes:
        mesh_name = meshes[mesh_idx].get("name", "?") if mesh_idx < len(meshes) else "?"
        print(f"  node[{idx}] name={name!r:30}  -> mesh[{mesh_idx}] ({mesh_name})")
    print()

    skins = data.get("skins", [])
    print(f"=== Skins ({len(skins)}) ===")
    for i, sk in enumerate(skins):
        bones = sk.get("joints", [])
        print(f"  [{i}] {sk.get('name', '?')}  joints={len(bones)}")
    print()

    animations = data.get("animations", [])
    print(f"=== Animations ({len(animations)}) ===")
    for i, anim in enumerate(animations):
        chans = anim.get("channels", [])
        print(f"  [{i}] {anim.get('name', '?')}  channels={len(chans)}")


if __name__ == "__main__":
    main()

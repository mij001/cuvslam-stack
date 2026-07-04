# docs/agent-memory/ — portable copy of the previous agent's file-memory

These are the memory notes the previous Claude agent kept in its private
per-project memory (`~/.claude/projects/.../memory/`), copied here so they
migrate with the repo to a new Claude account. **After cloning, read
`../../HANDOFF.md` first, then seed these into your own memory system** (write
each as a memory note; keep the frontmatter). Keep them updated as you work —
especially after any reboot (GPU clocks reset), power cut, or Tailscale re-auth.

| file | what |
|---|---|
| `MEMORY.md` | the index (one line per note) |
| `workstation-access.md` | ssh/sudo, the 575/CUDA-12.9/ncu-2025.2 unified stack, clock locks, free_gpu.zsh, datasets, setsid rule |
| `cuda-corruption-ld-library-path.md` | laptop CUDA is corrupt; LD_LIBRARY_PATH workaround until `sudo pacman -S cuda` |
| `cuvslam-stack-profiling-state.md` | where the research stands + what's next |

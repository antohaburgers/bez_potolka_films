# Polina v1 — asset storage

The source and reusable runtime assets are stored here, not rendered daily videos.
The approved v1 archive is prepared but its binary contents have NOT been uploaded yet.

Expected archive: `hockey/assets/polina/polina-v1-assets.zip`
SHA-256: `9ddcf13b6a24ae601a2abe79ff7f828b8655760e1e84debc361043b3495e93b7`
Size: 13523030 bytes.

After upload, verify from repository root:
```sh
echo '9ddcf13b6a24ae601a2abe79ff7f828b8655760e1e84debc361043b3495e93b7  hockey/assets/polina/polina-v1-assets.zip' | sha256sum -c -
unzip -t hockey/assets/polina/polina-v1-assets.zip
```

Inside:
- prototype_v05/ — approved base, original mouth v0.5, eyelids, prototype code, local test audio and phoneme timings; keep relative paths intact.
- references/ — energetic expression and transition, plus concept references; not yet aligned runtime overlays.
- manifest.json — SHA-256 of each resource and status.

The v0.5.1 mouth is rejected and excluded. Daily episode MP4s are excluded. Future individual layered assets can be committed as separate files instead of repeatedly replacing the archive. This public repository exposes uploaded assets. No publication is enabled. Cloud integration of the approved character remains a separate task.

#!/bin/bash
# Run ON the server:  bash inspect_job.sh <job_id>
# Registered images / points / reprojection error of every COLMAP sub-model of a job, and how many images
# the undistorted (trained-on) set contains.  The server trains on the LARGEST sub-model
# (not necessarily sparse/0) - compare "dense images" with the biggest "Registered images".
J=${GS_JOBS_DIR:-/data/jobs}/$1
echo "sparse models: $(ls $J/colmap/sparse | tr '\n' ' ')"
for d in $J/colmap/sparse/*; do
  echo "== $d"
  colmap model_analyzer --path $d 2>&1 | grep -E "Registered images|Points:|Mean track length|Mean reprojection" | sed 's/.*\] //'
done
echo "dense images (trained on): $(ls $J/colmap/dense/images | wc -l)"
echo "uploaded images:           $(ls $J/images | wc -l)"

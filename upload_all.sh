#!/bin/bash
while IFS='|' read -r local_name server_name; do
  echo "=== Uploading $local_name -> ${server_name}_prepared ==="
  gcloud compute scp "data_prepared/$local_name/transforms.json" "data_prepared/$local_name/points3d.ply" \
    "gs-training-server:/data/uploads/${server_name}_prepared/" \
    --zone=us-central1-a --project=dgs-training 2>&1 | tail -2
  gcloud compute scp --recurse "data_prepared/$local_name/images" \
    "gs-training-server:/data/uploads/${server_name}_prepared/" \
    --zone=us-central1-a --project=dgs-training 2>&1 | tail -2
  gcloud compute ssh gs-training-server --zone=us-central1-a --project=dgs-training \
    --command="echo -n 'images: '; ls /data/uploads/${server_name}_prepared/images | wc -l; ls /data/uploads/${server_name}_prepared/" 2>&1
  echo "=== Done: $server_name ==="
done < folder_map.txt
echo "ALL UPLOADS COMPLETE"

find ./content/raw_data -size 0 -exec rm {} +
find ./content/raw_data -type f ! -name "*.jpg" -exec rm {} +
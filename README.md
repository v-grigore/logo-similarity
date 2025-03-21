# Logo similarity - clustering domains based on logos

![alt text](https://github.com/v-grigore/logo-similarity/raw/master/docs/example.png)

Built a tool that extracts logos from a domains dataset, extracts visual features and clusters domains based on those features.

## How to use
Run with default dataset
```
python3 main.py
```
Run with custom dataset (must be `.parquet`; domains should be in `domain` column)
```
python3 main.py path/to/dataset
```
Save clustered images to `clustered_logos/`
```
python3 main.py --save
```
Ouput clusters saved to `data/output.db`

## Implementation details

### Step 1: Logo extraction
My first idea was to scrape the HTML pages to look for the links to logos and download each logo this way. This idea kind of worked, but scraping every single page was slow, and a lot of times
it didn't yield any logos due to unusual file structure within the server.

Eventually, I came across a free logo API (https://logo.clearbit.com/) with a 600 requests/min rate limit. This way I could extract ~2900 logos out of 3416 unique domains. Some domains do not exist
in the Clearbit database though, so we need to resort to other methods. I managed to extract ~380 more logos by trying to default to `domain/favicon.ico` and by scraping the HTML content for
specific `<link>` tags (e.g. `rel="shortcut icon"`).

For further optimization, I used a `ThreadPoolExecutor` with 10 workers, which resulted in ~100 logos/sec.  
Overall success rate: 96.14%.

![alt text](https://github.com/v-grigore/logo-similarity/raw/master/docs/logo_extraction_stats.png)

> __Note:__ Some domains did not work due to either inactive domains, bad redirection logic (e.g. no handler for HTTPS redirection) or wrong UserHeaders (some servers attempt to block bot-like
> connections). I got connection-related errors for ~80 websites. Accounting for broken websites ups success rate to 98.41%.

### Step 2: Logo recognition
The next step was implementing a way to extract important features from every logo, so that I could compare the logos between them. But before implementing feature extraction, I had to add
a preprocessing step so that each `.png` file had the same aspect ratio and resolution. I also had to remove alpha channel from images, because some extraction methods only work with "RGB" mode.
This meant I had to calculate average brigthness for each image, to determine whether to use black or white bg color (otherwise some of the all black/all white logos would be lost).

After preprocessing images, I had multiple ways of implementing feature extraction. First methods I tried were SIFT and ORB algorithms, which resulted in a pretty mediocre feature extraction, but
a really bad performance. Then I decided to try histogram extraction, assuming logos were simple enough for color distribution comparison. The histogram method yielded terrible results however, so
I decided to try a CNN model. Since I didn't have a large enough dataset, and training a model specifically for logo recogntion would take a lot of time and resources, I decided to use a
pre-trained model.

After trying ResNet50 and MobileNetV2 pre-trained models, I decided to stick with MobileNetV2, which produced slightly better results while also being a little bit faster (~6 logos/sec vs ~4
logos/sec). Saved extracted features to `features.npz` file to spare time while testing clustering methods.

### Step 3: Logo clustering
Now that I had the features, the final step was clustering all the domains based on their resulting logo features. Since I didn't know how many unique groups there were, I decided to use DBSCAN.
For early testing, DBSCAN performed fine, especially when I was still experimenting with different feature extraction methods.

One big problem was that the data clusters have varying densities, which meant that simple parameter tuning would not suffice. A big improvement was switching to HDBSCAN, which handled varying
density clusters a lot better, but it still wasn't good enough. I came up with the idea to run multiple HDBSCANs with less and less restrictive parameters, this way I managed to group large
amounts of very similar logos in the first iteration of HDBSCAN, and then I grouped small amounts of somewhat similar logos in the second iteration.

I also had to perform PCA to reduce high dimensionality for better distance comparison. Finally, I implemented a display function using plotly, to better visualize the different clusters:

![alt text](https://github.com/v-grigore/logo-similarity/raw/master/docs/clustered_data_points.png)

> __Note:__ Noise points not shown on graph.

## Result
Obtained over 300 clusters and 900 noise points (single domain groups). The majority of logos were correctly grouped together (especially the large groups of near-identical logos), but the program
failed sometimes to identify small groups of similar logos (2-3 logos with slight variations, for instance). Some of the smaller clusters were merged together, despite belonging to different
groups. The logo extraction wasn't perfect either, mainly due to outdated logos in the APIs database. Overall, I am satisfied with the obtained results.

## Next steps
1. Switch to a paid logo API for better extraction (Clearbit's legacy API will shut down on 1-Dec-2025)
2. Train a specialized CNN model for logo recognition
3. Use the obtained results to build a training dataset for a classification model

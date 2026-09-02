# embeddings-visualizer
Visualize vectors in an index in a latent space.

Prototype of a visual (graphic) interface for an index.

1. API call to vector store to get vectors in an index. (May visualize all or a subset of them.)
   - extract the metadata
     - additional_info contains embedding model information ?
   - how to pass auth token ? (for now just using the one for the one index shown?)
2. Visual:
   - grid/graph background, Eluvio/EVIE color scheme (purple?)
   - each vector as a node (arrow with magnitude 1?)
   - hover over a node highlights it and shows the vector's formatted metadata
     - if it is embedded text, show the text too
   - show "Embedded with ___ model" in a corner?
   - different colors for different modality embeddings (vector for text vs. image vs. video)? with a key/legend in the corner?
3. Search:
   - search box in the same corner as "Embedded with ..."?
   - search query (text or image?) is embedded and inserted as a new node in the same latent space (different color?)
     - need to check if embedding model supports text and image query modalities? if text-only, then show Error message?
4. Media content:
   - if a vector is an embedded photo or bounding box or video, show the media content (retrieve from fabric - same auth question as above?)
5. May be later integrated/rendered in EVIE? --> if in EVIE, then does tenant automatically have access (auth)?


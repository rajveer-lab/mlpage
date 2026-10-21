# MSR Cambridge traces (not included)

The MSR Cambridge block I/O traces (Narayanan et al., FAST 2008) are distributed
by the SNIA IOTTA repository under SNIA's trace licence, so they are not
committed here.

1. Download one or more volumes from http://iotta.snia.org/traces/block-io/388
   (for example `web_0.csv.gz`, `prxy_0.csv.gz`, `hm_0.csv.gz`). A few volumes
   are enough; the full first set is about 3 GB.
2. Put the `.csv` or `.csv.gz` files in this folder.
3. `python -m mlpr evaluate` picks them up automatically as extra unseen test
   traces (the first 200,000 page references of each, 4 KB pages).

To look at one trace on its own:

    python -m mlpr simulate --msr data/msr/web_0.csv.gz --memory 0.05

IOTTA notes that older block traces may not reflect modern systems, which is
why the synthetic and program-derived workloads are the main test set.

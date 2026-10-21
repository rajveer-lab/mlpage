import gzip

from mlpr.metrics import fault_reduction, gap_closed
from mlpr.traces import bfs_trace, load_msr, load_trace_file, matmul_trace, mergesort_trace
from mlpr.workloads import frames_for, load_split, memory_sizes


def test_program_traces_are_deterministic_and_collapsed():
    for make in (lambda: matmul_trace(12), lambda: mergesort_trace(256, seed=1), lambda: bfs_trace(300, seed=1)):
        a, b = make(), make()
        assert a == b and len(a) > 100
        assert all(x != y for x, y in zip(a, a[1:]))  # consecutive repeats collapsed


def test_matmul_touches_three_matrices():
    n, page_bytes = 16, 64
    pages_per_matrix = n * n * 8 // page_bytes
    assert len(set(matmul_trace(n, page_bytes))) == 3 * pages_per_matrix


def test_splits_differ():
    train, test = load_split("train"), load_split("test", with_msr=False)
    assert train.keys() == test.keys()
    assert all(train[name] != test[name] for name in train)


def test_memory_sizes():
    trace = list(range(1000))
    assert frames_for(trace, 0.05) == 50
    assert frames_for(trace, 0.001) == 8
    assert [f for _, f in memory_sizes(list(range(300)))] == [8, 15, 30]  # 1% and 2% both floor to 8


def test_msr_loader(tmp_path):
    rows = [
        "128166372003061629,web,0,Read,8192,4096,100",
        "128166372003061630,web,0,Write,4096,8192,100",
        "128166372003061631,web,0,Read,8192,1,100",
    ]
    path = tmp_path / "web_0.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(rows) + "\n")
    # pages 2 | 1, 2 | 2  -> renumbered in order of first use
    assert load_msr(path) == [0, 1, 0, 0]
    assert load_msr(path, op="Read") == [0, 0]


def test_trace_file(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("# a comment\n1 2 3\n4,5\n")
    assert load_trace_file(path) == [1, 2, 3, 4, 5]


def test_formulas():
    assert fault_reduction(100, 80) == 20.0
    assert gap_closed(100, 80, 60) == 50.0

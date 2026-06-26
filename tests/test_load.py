import os

import numpy as np
import pytest

from load import PARAMETERS, load_partition, parse_instance

PARTITION3 = os.path.join("data", "partition3_instances.tar.gz")


def test_parameters_has_24_names():
    assert len(PARAMETERS) == 24
    assert PARAMETERS[0] == "TOTUSJH"
    assert PARAMETERS[-1] == "R_VALUE"


def test_parse_instance_reads_values_and_marks_missing():
    header = "Timestamp\t" + "\t".join(PARAMETERS)
    row1 = "2013-11-01 20:00:00\t" + "\t".join(str(float(i)) for i in range(24))
    row2 = "2013-11-01 20:12:00\t" + "\t".join(["None"] * 24)
    raw = "\n".join([header, row1, row2]).encode("utf-8")
    arr = parse_instance(raw)
    assert arr.shape == (2, 24)
    assert arr[0, 0] == 0.0
    assert arr[0, 23] == 23.0
    assert np.isnan(arr[1]).all()


@pytest.mark.skipif(not os.path.exists(PARTITION3), reason="partition3 not downloaded")
def test_load_partition3_smoke():
    features, labels, classes, names = load_partition(PARTITION3, limit=50)
    assert len(features) == 50
    assert features[0].shape[1] == 24
    assert set(np.unique(labels)).issubset({0, 1})
    assert all(f.ndim == 2 for f in features)

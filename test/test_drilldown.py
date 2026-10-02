import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP = PROJECT_ROOT / "src" / "week9_streamlit_starter.py"
CUBE = PROJECT_ROOT / "data" / "processed_data_cube.csv"


def _decode_y(trace):
    y_enc = trace["y"]
    return np.frombuffer(base64.b64decode(y_enc["bdata"]),
                         dtype=np.dtype(y_enc["dtype"]))


def _drilldown_spec(at):
    return json.loads(at.get("plotly_chart")[-1].spec)


def test_drilldown_renders_and_reacts_to_pickers():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception

    # Defaults are the first option of each dict.
    assert [sb.label for sb in at.selectbox] == ["Dimension", "Metric"]
    assert at.selectbox[0].value == "Contract type"
    assert at.selectbox[1].value == "Total applications"
    assert len(at.get("plotly_chart")) == 2  # contract-type chart + drilldown

    at.selectbox[0].select("Age group")
    at.selectbox[1].select("Total defaults")
    at.run()
    assert not at.exception

    spec = _drilldown_spec(at)
    trace = spec["data"][0]
    assert spec["layout"]["title"]["text"] == "Total defaults by Age group"
    assert trace["x"][0] == "26-35"  # largest category in the data
    ys = _decode_y(trace).tolist()
    assert ys == sorted(ys, reverse=True)  # bars sorted descending


def test_drilldown_respects_month_filter():
    # Expected values straight from the cube: 2023 defaults by age group.
    cube = pd.read_csv(CUBE)
    in_2023 = cube[cube["YEAR_APPLIED"] == 2023]
    expected = in_2023.groupby("AGE_GROUP")["total_defaults"].sum().sort_values(ascending=False)

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.selectbox[0].select("Age group")
    at.selectbox[1].select("Total defaults")
    at.slider[0].set_range(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-01"))
    at.run()
    assert not at.exception

    trace = _drilldown_spec(at)["data"][0]
    assert trace["x"] == list(expected.index)
    assert _decode_y(trace).tolist() == expected.tolist()

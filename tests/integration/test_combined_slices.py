"""Cross-slice checks on the real server registration/filter wrapper.

Integration branch only: existing upstream PRs stay independent. Network calls
are mocked; local notebook records use temporary directories and synthetic FIT.
"""

import json

import pytest
from mcp.server.fastmcp import FastMCP

import garmin_mcp
from garmin_mcp import tool_profiles
from scripts.generate_dxt_tools import MODULES
from tests.fixtures.synthetic_fit import activity_fit

NEW_TOOLS = {
    "get_activity_resampled_stream",
    "preview_cycling_workout",
    "record_physiology_observation",
    "list_physiology_observations",
}


def registered_app(client, profile=None):
    app = FastMCP("Combined slices")
    proxy = garmin_mcp._ToolFilter(app, set(), set(), profile=profile)
    for module in MODULES:
        module.configure(client)
        module.register_tools(proxy)
    assert proxy.unannotated_tool_names() == []
    return app


async def invoke(app, tool_name, **arguments):
    result = await app.call_tool(tool_name, arguments)
    blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text)


@pytest.mark.asyncio
async def test_combined_real_filter_annotations(mock_garmin_client):
    app = registered_app(mock_garmin_client)
    tools = {tool.name: tool for tool in await app.list_tools()}
    assert NEW_TOOLS <= tools.keys()
    for name in NEW_TOOLS:
        assert tools[name].outputSchema is None  # no doubled JSON wrapper
    assert tools["get_activity_resampled_stream"].annotations.readOnlyHint is True
    assert tools["get_activity_resampled_stream"].annotations.openWorldHint is True
    for name in NEW_TOOLS - {"get_activity_resampled_stream"}:
        assert tools[name].annotations.openWorldHint is False
    assert tools["record_physiology_observation"].annotations.readOnlyHint is False
    assert tools["record_physiology_observation"].annotations.idempotentHint is False
    assert mock_garmin_client.mock_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile,expected",
    [
        (tool_profiles.MINIMAL_TOOLS, set()),
        (tool_profiles.ACTIVITIES_TOOLS, {"get_activity_resampled_stream"}),
        (tool_profiles.WORKOUTS_TOOLS, {"preview_cycling_workout"}),
        (
            tool_profiles.TRAINING_TOOLS,
            {"record_physiology_observation", "list_physiology_observations"},
        ),
    ],
)
async def test_new_tools_do_not_bleed_between_profiles(
    mock_garmin_client, profile, expected
):
    app = registered_app(mock_garmin_client, profile)
    names = {t.name for t in await app.list_tools()}
    assert NEW_TOOLS & names == expected
    assert mock_garmin_client.mock_calls == []


@pytest.mark.asyncio
async def test_real_fit_decode_through_server_wrapper(mock_garmin_client):
    app = registered_app(mock_garmin_client)
    mock_garmin_client.download_activity.return_value = activity_fit(duration=5)
    data = await invoke(
        app,
        "get_activity_resampled_stream",
        activity_id=42,
        fields=["hr", "power"],
        resolution="5s",
    )
    assert data["status"] == "ok"
    assert data["data"][0]["power"] == 180
    assert data["summary"]["total_active_s"] == 5
    mock_garmin_client.upload_workout.assert_not_called()
    mock_garmin_client.client.post.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_local_to_upload_to_schedule_flow(
    mock_garmin_client, tmp_path, monkeypatch
):
    """Each boundary is an explicit tool call, not new automatic orchestration."""
    app = registered_app(mock_garmin_client)
    directory = tmp_path / "private"
    monkeypatch.setenv("GARMIN_DATA_DIR", str(directory))
    observation = {
        "sport": "cycling",
        "metric": "lthr",
        "value": 160,
        "unit": "bpm",
        "observed_at": "2024-01-01T12:00:00Z",
        "source": "Synthetic user report",
    }
    proposed = await invoke(
        app, "record_physiology_observation", observation=observation
    )
    assert proposed["status"] == "preview" and not directory.exists()
    saved = await invoke(
        app, "record_physiology_observation", observation=observation, dry_run=False
    )
    listed = await invoke(
        app, "list_physiology_observations", sport="cycling", metric="lthr"
    )
    assert listed["observations"] == [saved["observation"]]
    ride = await invoke(
        app,
        "preview_cycling_workout",
        name="Synthetic ride",
        steps=[{"duration_seconds": 300, "hr_zone": "Z2"}],
    )
    assert ride["write_performed"] is False
    assert mock_garmin_client.mock_calls == []

    mock_garmin_client.upload_workout.return_value = {"workoutId": 42}
    uploaded = await invoke(app, "upload_workout", workout_data=ride["workout_data"])
    assert uploaded["workout_id"] == 42
    mock_garmin_client.upload_workout.assert_called_once()
    mock_garmin_client.client.post.assert_not_called()

    mock_garmin_client.query_garmin_graphql.return_value = {
        "data": {"workoutScheduleSummariesScalar": []}
    }
    response = mock_garmin_client.client.post.return_value
    response.status_code = 200
    response.json.return_value = {"workoutScheduleId": 701, "workoutId": 42}
    scheduled = await invoke(
        app, "schedule_workout", workout_id=42, calendar_date="2024-01-02"
    )
    assert scheduled["scheduled_workout_id"] == "701"
    mock_garmin_client.client.post.assert_called_once_with(
        "connectapi", "workout-service/schedule/42", json={"date": "2024-01-02"}
    )
    unscheduled = await invoke(app, "unschedule_workout", scheduled_workout_id=701)
    assert unscheduled["status"] == "success"
    mock_garmin_client.unschedule_workout.assert_called_once_with(701)

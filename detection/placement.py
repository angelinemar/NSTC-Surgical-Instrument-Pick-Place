"""Retry rejected physical arrangements, never rendering or labeling failures."""

class ScenePlacementError(RuntimeError):
    pass


def retry_placement(prepare,report_failure,maximum_attempts=5):
    for attempt in range(maximum_attempts):
        try:
            return prepare(attempt)
        except ScenePlacementError as error:
            report_failure(attempt,error)
            if attempt+1==maximum_attempts:
                raise ScenePlacementError(f'No valid placement after {maximum_attempts} attempts: {error}') from error

import pathlib
import re


def test_eloquent_torch2trt_combines_apt_update_and_install():
    dockerfile = pathlib.Path("docker/dockerfile.ros.eloquent.torch2trt")
    assert dockerfile.exists()
    content = dockerfile.read_text()
    # The old split-pattern must be gone so Docker cannot cache a stale update layer.
    assert "RUN apt-get update\nRUN apt-get install" not in content
    # The final apt section must combine update, install, and cleanup in one RUN.
    assert re.search(
        r"RUN apt-get update\s*&&\s*apt-get install.*?&&\s*rm -rf /var/lib/apt/lists/\*",
        content,
        re.DOTALL,
    )

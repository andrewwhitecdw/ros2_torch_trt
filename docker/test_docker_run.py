# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

import os


def test_docker_run_uses_correct_pwd_variable():
    script_path = os.path.join(os.path.dirname(__file__), 'docker_run.sh')
    with open(script_path) as f:
        script = f.read()

    assert '${pwd}' not in script, 'docker_run.sh should not use ${pwd}'
    assert '$(pwd)' in script or '$PWD' in script, 'docker_run.sh should use $(pwd) or $PWD'

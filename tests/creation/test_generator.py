import pytest
from rail.creation import Generator


def test_Generator_instantiation():
    with pytest.raises(TypeError):
        Generator()
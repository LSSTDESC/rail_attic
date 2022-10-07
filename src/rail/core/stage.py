""" Base class for PipelineStages in Rail """

import os

from ceci import PipelineStage, MiniPipeline
from ceci.stage import StageBuilder
from ceci.config import StageParameter as Param


class RailPipeline(MiniPipeline):
    """A pipeline intended for interactive use

    Mainly this allows for more concise pipeline specification, along the lines of:

    self.stage_1 = Stage1Class.build(...)
    self.stage_2 = Stage2Class.build(connections=dict(input=self.stage1.io.output), ...)

    And end up with a fully specified pipeline.
    """

    def __init__(self):
        MiniPipeline.__init__(self, [], dict(name='mini'))

    def __setattr__(self, name, value):
        if isinstance(value, StageBuilder):
            stage = value.build(name)
            self.add_stage(stage)
            return stage
        return MiniPipeline.__setattr__(self, name, value)


class RailStage(PipelineStage):
    pass

import amrlib
import penman


def post_processing(string: str, **kwargs) -> str:
    model_path = kwargs["model_path"]
    stog = amrlib.load_stog_model(model_path)
    amr_graph = stog.parse_sents([string])[0]
    return amr_graph

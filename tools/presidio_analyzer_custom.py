import gradio as gr
from typing import List, Iterable, Dict, Union, Any, Optional, Iterator, Tuple
from tqdm import tqdm

from presidio_analyzer import DictAnalyzerResult, RecognizerResult #, AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpArtifacts



def analyze_iterator_custom(
        self,
        texts: Iterable[Union[str, bool, float, int]],
        language: str,
        list_length:int,
        progress=gr.Progress(),
        **kwargs,
    ) -> List[List[RecognizerResult]]:
        """
        Analyze an iterable of strings.

        :param texts: An list containing strings to be analyzed.
        :param language: Input language
        :param list_length: Length of the input list.
        :param kwargs: Additional parameters for the `AnalyzerEngine.analyze` method.
        """

        # validate types
        texts = self._validate_types(texts)

        # Process the texts as batch for improved performance
        nlp_artifacts_batch: Iterator[
            Tuple[str, NlpArtifacts]
        ] = self.analyzer_engine.nlp_engine.process_batch(
            texts=texts, language=language
        )

        

        list_results = []

        # Uncomment this if you want to show progress within a file
        #for text, nlp_artifacts in progress.tqdm(nlp_artifacts_batch, total = list_length, desc = "Analysing text for personal information", unit = "rows"):
        for text, nlp_artifacts in nlp_artifacts_batch:
            results = self.analyzer_engine.analyze(
                text=str(text), nlp_artifacts=nlp_artifacts, language=language, **kwargs
            )

            list_results.append(results)

        return list_results

def analyze_dict(
        self,
        input_dict: Dict[str, Union[Any, Iterable[Any]]],
        language: str,
        keys_to_skip: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterator[DictAnalyzerResult]:
        """
        Analyze a dictionary of keys (strings) and values/iterable of values.

        Non-string values are returned as is.

        :param input_dict: The input dictionary for analysis
        :param language: Input language
        :param keys_to_skip: Keys to ignore during analysis
        :param kwargs: Additional keyword arguments
        for the `AnalyzerEngine.analyze` method.
        Use this to pass arguments to the analyze method,
        such as `ad_hoc_recognizers`, `context`, `return_decision_process`.
        See `AnalyzerEngine.analyze` for the full list.
        """

        context = []
        if "context" in kwargs:
            context = kwargs["context"]
            del kwargs["context"]

        if not keys_to_skip:
            keys_to_skip = []

            
        for key, value in input_dict.items():
            if not value or key in keys_to_skip:
                yield DictAnalyzerResult(key=key, value=value, recognizer_results=[])
                continue  # skip this key as requested

            # Add the key as an additional context
            specific_context = context[:]
            specific_context.append(key)

            if type(value) in (str, int, bool, float):
                results: List[RecognizerResult] = self.analyzer_engine.analyze(
                    text=str(value), language=language, context=[key], **kwargs
                )
            elif isinstance(value, dict):
                new_keys_to_skip = self._get_nested_keys_to_skip(key, keys_to_skip)
                results = self.analyze_dict(
                    input_dict=value,
                    language=language,
                    context=specific_context,
                    keys_to_skip=new_keys_to_skip,
                    **kwargs,
                )
            elif isinstance(value, Iterable):
                # Recursively iterate nested dicts
                list_length = len(value)

                results: List[List[RecognizerResult]] = analyze_iterator_custom(self,
                    texts=value,
                    language=language,
                    context=specific_context,
                    list_length=list_length,
                    **kwargs,
                )
            else:
                raise ValueError(f"type {type(value)} is unsupported.")

            yield DictAnalyzerResult(key=key, value=value, recognizer_results=results)
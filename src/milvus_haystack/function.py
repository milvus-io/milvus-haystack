import uuid
from typing import Any, Dict, List, Optional, Union

from haystack import default_from_dict, default_to_dict
from pymilvus import Function, FunctionType

from milvus_haystack.utils.constant import SPARSE_VECTOR_FIELD, TEXT_FIELD


class BaseMilvusBuiltInFunction:
    """
    Base class for Milvus built-in functions.

    See:
    https://milvus.io/docs/manage-collections.md#Function
    """

    def __init__(self) -> None:
        self._function: Optional[Function] = None

    @property
    def function(self) -> Function:
        return self._function

    @property
    def input_field_names(self) -> Union[str, List[str]]:
        return self.function.input_field_names

    @property
    def output_field_names(self) -> Union[str, List[str]]:
        return self.function.output_field_names

    @property
    def type(self) -> FunctionType:
        return self.function.type


class BM25BuiltInFunction(BaseMilvusBuiltInFunction):
    """
    Milvus BM25 built-in function.

    See:
    https://milvus.io/docs/full-text-search.md
    """

    def __init__(
        self,
        *,
        input_field_names: str = TEXT_FIELD,
        output_field_names: str = SPARSE_VECTOR_FIELD,
        analyzer_params: Optional[Dict[Any, Any]] = None,
        enable_match: bool = False,
        function_name: Optional[str] = None,
    ):
        """
        Args:
            input_field_names (str): The name of the input field, default is 'text'.
            output_field_names (str): The name of the output field, default is 'sparse'.
            analyzer_params (Optional[Dict[Any, Any]]): The parameters for the analyzer.
                Default is None. See:
                https://milvus.io/docs/analyzer-overview.md#Analyzer-Overview
            enable_match (bool): Whether to enable match.
            function_name (Optional[str]): The name of the function. Default is None,
                which means a random name will be generated.
        """
        super().__init__()
        if function_name:
            self._function_name = function_name
        else:
            self._function_name = f"bm25_function_{str(uuid.uuid4())[:8]}"
        self._input_field_names = input_field_names
        self._output_field_names = output_field_names

        self._function = Function(
            name=self._function_name,
            input_field_names=input_field_names,
            output_field_names=output_field_names,
            function_type=FunctionType.BM25,
        )
        self.analyzer_params: Optional[Dict[Any, Any]] = analyzer_params
        self.enable_match = enable_match

    def get_input_field_schema_kwargs(self) -> Dict:
        """
        Get the input field schema kwargs for the function.

        """
        field_schema_kwargs: Dict[Any, Any] = {
            "enable_analyzer": True,
            "enable_match": self.enable_match,
        }
        if self.analyzer_params is not None:
            field_schema_kwargs["analyzer_params"] = self.analyzer_params
        return field_schema_kwargs

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the function to a dictionary.

        """

        init_parameters = {
            "function_name": self._function_name,
            "input_field_names": self._input_field_names,
            "output_field_names": self._output_field_names,
            "analyzer_params": self.analyzer_params,
            "enable_match": self.enable_match,
        }
        return default_to_dict(self, **init_parameters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25BuiltInFunction":
        """
        Deserialize the function from a dictionary.

        """
        return default_from_dict(cls, data)

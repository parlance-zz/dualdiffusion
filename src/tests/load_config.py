# MIT License
#
# Copyright (c) 2023 Christopher Friesen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from utils import config

from typing import Optional
from dataclasses import dataclass
from pprint import pprint

from utils.dual_diffusion_utils import dict_str


@dataclass
class TestSubClass:
    my_str: str = "abcd"
    my_int: int = 5
    my_float: float = 10.0

@dataclass
class TestConfig:
    my_dict: dict[str, str]

    my_str_list: list[str] 
    my_int_list: list[int]
    my_float_list: list[int]
    my_dict_list: list[dict[str, str]]

    my_str_dict: dict[str, str]
    my_int_dict: dict[str, int]
    my_float_dict: dict[str, float]
    my_dict_dict: dict[str, dict[str, str]]

    my_data_class: TestSubClass
    my_data_class_list: list[TestSubClass]
    my_data_class_dict: dict[str, TestSubClass]

    my_str: str = "abcd"
    my_int: int = 5
    my_float: float = 10.0

if __name__ == "__main__":

    test_data = {}
    test_data["my_dict"] = {"abc": "123"}
    test_data["my_str_list"] = ["a", "b", "c"]
    test_data["my_int_list"] = [1,2,3]
    test_data["my_float_list"] = [4.,5.,6.]
    test_data["my_dict_list"] = [{"abc": "123"}, {"def": "456"}]

    test_data["my_str_dict"] = {"a": "d", "b": "e", "c": "f"}
    test_data["my_int_dict"] = {"a": 1, "b": 2, "c": 3}
    test_data["my_float_dict"] = {"a": 4., "b": 5., "c": 6.}
    test_data["my_dict_dict"] = {"x": {"abc": "123"}, "y": {"def": "456"}}

    test_data["my_data_class"] = {**TestSubClass().__dict__}
    test_data["my_data_class_list"] = [{**test_data["my_data_class"]}, {**test_data["my_data_class"]}]
    test_data["my_data_class_dict"] = {"a": {**test_data["my_data_class"]}, "b": {**test_data["my_data_class"]}}

    pprint(config.load_config(TestConfig, "fake_path.json", test_data))
:- module(utils, [get_dict_from_json_file/2]).
:- use_module(library(http/json)).

get_dict_from_json_file(FPath, Dicty) :-
  open(FPath, read, Stream), json_read_dict(Stream, Dicty).

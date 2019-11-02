:- module(utils, [get_dict_from_json_file/2, get_lower_upper_bounds_from_list/3]).
:- use_module(library(http/json)).

get_dict_from_json_file(Dicty, FPath) :-
  open(FPath, read, Stream), json_read_dict(Stream, Dicty).

get_lower_upper_bounds_from_list(Lower, Upper, List) :-
  nth0(0, List, Lower),
  nth0(1, List, Upper).

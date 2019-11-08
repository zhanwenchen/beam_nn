:- module(utils, [get_dict_from_json_file/2,
                  get_lower_upper_bounds_from_list/3,
                  get_output_size/3,
                  is_network_legal/2]).
:- use_module(library(http/json)).

get_dict_from_json_file(Dicty, FPath) :-
  open(FPath, read, Stream), json_read_dict(Stream, Dicty), close(Stream).

get_lower_upper_bounds_from_list(Lower, Upper, List) :-
  nth0(0, List, Lower),
  nth0(1, List, Upper).

% For conv1d, set all heights to 1 so that we have a record.
get_output_size([InputHeight, InputWidth, _], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  % Read variables from dictionary
  CurrentLayer.type == conv1d,
  OutChannels is CurrentLayer.out_channels,
  CurrentLayer.kernel_height == 1,
  KernelWidth is CurrentLayer.kernel_width,
  % writeln(CurrentLayer.padding_height),
  CurrentLayer.padding_height == 0,
  CurrentLayer.stride_height == 0,
  % writeln(CurrentLayer.kernel_height),
  % PaddingHeight is 1,
  PaddingWidth is CurrentLayer.padding_width,
  % StrideHeight is 1,
  StrideWidth is CurrentLayer.stride_width,

  OutputHeight is 1,
  OutputWidth is floor((InputWidth - KernelWidth + 2 * PaddingWidth) / StrideWidth + 1),
  OutputDepth is OutChannels.

get_output_size([InputHeight, InputWidth, _], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == conv2d,
  OutChannels is CurrentLayer.out_channels,
  KernelHeight is CurrentLayer.kernel_height,
  KernelWidth is CurrentLayer.kernel_width,
  PaddingHeight is CurrentLayer.padding_height,
  PaddingWidth is CurrentLayer.padding_width,
  StrideHeight is CurrentLayer.stride_height,
  StrideWidth is CurrentLayer.stride_width,

  OutputHeight is floor((InputHeight - KernelHeight + 2 * PaddingHeight) / StrideHeight + 1),
  OutputWidth is floor((InputWidth - KernelWidth + 2 * PaddingWidth) / StrideWidth + 1),
  OutputDepth is OutChannels.

get_output_size([InputHeight, InputWidth, InputDepth], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == upsample,
  ScaleFactorHeight is CurrentLayer.scale_factor_height,
  ScaleFactorWidth is CurrentLayer.scale_factor_width,

  OutputHeight is ScaleFactorHeight * InputHeight,
  OutputWidth is ScaleFactorWidth * InputWidth,
  OutputDepth is InputDepth.

get_output_size([InputHeight, InputWidth, InChannels], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == maxpool1d,
  KernelHeight is 1,
  KernelWidth is CurrentLayer.kernel_width,
  StrideHeight is 1,
  StrideWidth is CurrentLayer.stride_width,

  OutputHeight is 1,
  OutputWidth is floor((InputWidth - KernelWidth) / StrideWidth + 1),
  OutputDepth is InChannels.

get_output_size([InputHeight, InputWidth, InChannels], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == maxpool2d,
  KernelHeight is CurrentLayer.kernel_height,
  KernelWidth is CurrentLayer.kernel_width,
  StrideHeight is CurrentLayer.stride_height,
  StrideWidth is CurrentLayer.stride_width,

  OutputHeight is floor((InputHeight - KernelHeight) / StrideHeight + 1),
  OutputWidth is floor((InputWidth - KernelWidth) / StrideWidth + 1),
  OutputDepth is InChannels.


get_output_size([_, _, InChannels], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == adaptiveavgpool2d,

  OutputHeight is CurrentLayer.out_height,
  OutputWidth is CurrentLayer.out_width,
  OutputDepth is InChannels.

is_network_legal(_, []).
is_network_legal(_, [CurrentLayer]) :-
  CurrentLayer.type == fcs.
is_network_legal(InputSizes, [CurrentLayer|RestLayers]) :-
  get_output_size(InputSizes, CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]),
  OutputHeight >= 1,
  OutputWidth >= 1,
  integer(OutputDepth), OutputDepth >= 1,
  is_network_legal([OutputHeight, OutputWidth, OutputDepth], RestLayers).

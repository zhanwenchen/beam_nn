%% create_models.pl
%% usage: Under project directory (containing this file, create.pl)
%% $ swipl
%% ?- ['create.pl']
%% ?- main(50) % create 50 models (under project/DNNs/)
%% v1.0: Implemented FCN
%% v1.1: Implemented FCN with upsample layer dictionaries.
%% v1.2: Fixed conv1d height (kernel, stride, padding) > 1 issue.
%% v1.3: 1. Conv1, Conv2 now allow stride = 2. Conv3, Conv4 still force stride=1; 2. (1.) stride change enables normal conv kernel sizes between 3 and 8.
%% v1.4: 1. Fixed conv2d stride height = 0 issue; 2. Narrowed conv2d stride height from [1, 3] to [1, 2]; 3. Narrowed conv2d padding height from [0, 3] to [0, 2].
%% v1.5: Narrowed kernel height to 1 (conv1d) and [2, 3] (conv2d).
%% v1.6.0: 1. Authoritative change to conv kernel widths: between [3, 15]. 2 is too small, and 15 is pretty big already. 2. Implemented get_dict_from_json_file to read in JSON Prolog dicts. 3. Refactored get_dict_from_json_file to utils.pl module. 4. Integrated hyperparam_ranges_fcn.json file to parameterize search ranges (partially).
%  v1.6.1 Remove default LeakyReLU after last layer because regression should not have nonlinear output activation. Also added batch norm.
%  v1.6.2 Narrowed learning rate to either 1e-04 or 1e-05.
%  v1.6.3 Reduced patience from 30 to 20 in order to churn out more models
%  v1.6.4 Use new dataset - reject-only point targets.
%  v1.6.5 Use new dataset - smooth-only point targets.
%  v1.6.6 Non-random everything except for kernel sizes.
%% TODO: combine conv1d and conv2d: it's just a matter of height=1.
%% BUG: Potential mixup between random_member and random_between.
:- use_module(library(http/json)).
:- use_module(lib/utils, [get_dict_from_json_file/2,
                          get_lower_upper_bounds_from_list/3,
                          get_output_size/3,
                          is_network_legal/2]).
:- set_prolog_flag(verbose, silent).
:- initialization(main).


% Predicate to find fully-convolutional networks
find_architecture_mlpb5(MLPB5, InputSize) :-
  % repeat, % So that random_between can run more than once.

  % FC1 sizes
  FC1Width is 1024,
  FC1 = fc1{name: fc1, type: fc, in_channels: InputSize, out_channels: FC1Width},

  FC2Width is 1024,
  FC2 = fc2{name: fc2, type: fc, in_channels: FC1Width, out_channels: FC2Width},

  % FC3 is bottleneck
  random_member(FC3Width, [32, 64, 128, 256, 512]),
  writeln(FC3Width),
  FC3 = fc3{name: fc3, type: fc, in_channels: FC2Width, out_channels: FC3Width},

  FC4Width is 1024,
  FC4 = fc4{name: fc4, type: fc, in_channels: FC3Width, out_channels: FC4Width},

  FC5 = fc5{name: fc5, type: fc, in_channels: FC4Width, out_channels: InputSize},

  MLPB5 = [FC1, FC2, FC3, FC4, FC5].

% model layers and training stuff
find_training_hyperparams_MLPB5(MLPB5) :-
  % get_dict_from_json_file(ModelParamsRangesDict, 'hyperparam_ranges_fcn.json'),

  InputSize = 130,

  find_architecture_mlpb5(Layers, InputSize),

  LossFunction = 'MSE',
  Optimizer = 'Adam',

  LearningRate is 1E-5,

  WeightDecay is 0,

  % Training and validation data locations
  NumScatter is 1,
  % TODO: switchable training data.
  DataDirname = 'data/20180402_L74_70mm',
  atomic_list_concat([DataDirname, '/train_', NumScatter, '.h5'], DataTrain),
  atomic_list_concat([DataDirname, '/val_', NumScatter, '.h5'], DataVal),

  Momentum is 0,

  Version = '0.1.0',

  MLPB5 = model{model: 'MLPB5',
              % input_dims: InputDims,]
              dropout: 0.5,
              input_size: InputSize,
              version: Version,
              loss_function: LossFunction,
              optimizer: Optimizer,
              learning_rate: LearningRate,
              weight_decay: WeightDecay,
              data_is_target: 0,
              batch_size: 32,
              data_noise_gaussian: 1,
              patience: 20,
              data_train: DataTrain,
              data_val: DataVal,
              momentum: Momentum,
              layers: Layers}.

write_model_to_file_per_k(Dict, Dirname, K) :-
  atomic_list_concat([Dirname, '/', 'k_', K], Kname),
  make_directory(Kname),
  atomic_list_concat([Kname, '/', 'model_params.json'], Fname),
  NewDict = Dict.put(k, K),
  open(Fname, write, Stream),
  json_write_dict(Stream, NewDict),
  close(Stream).

write_model_to_file(Dict) :-
  Version = '0.1.0',
  timestring(Timestring),
  atomic_list_concat(['mlpb5_', Version, '_', Timestring, '_created'], ModelName),
  atomic_list_concat(['DNNs/', ModelName], Dirname),
  make_directory(Dirname),
  NewDict = Dict.put(name, ModelName),
  maplist(write_model_to_file_per_k(NewDict, Dirname), [3, 4, 5]),
  writeln(Dirname).

timestring(Timestring) :-
  get_time(Timestamp), format_time(atom(Timestring), '%Y%m%d%H%M%S%f', Timestamp).

find_and_write_mlpb5 :-
  find_training_hyperparams_MLPB5(MLPB5), write_model_to_file(MLPB5).

main :-
  current_prolog_flag(argv, [HowManyStr|_]),
  atom_number(HowManyStr, HowManyInt),
  once(findnsols(HowManyInt, [], (repeat, find_and_write_mlpb5), _)),
  halt.

% main(HowMany) :-
%   once(findnsols(HowMany, [], (repeat, find_and_write_fcn), Y)).

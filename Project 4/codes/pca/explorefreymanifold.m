function explorefreymanifold(Y,X)
% explorefreymanifold - interactively view images from manifold position
% EXPLOREFREYMANIFOLD(Y,X) plots Y(1,:) against Y(2,:) and then
%   enters an interactive mode to allow users to click on or near a
%   point and see the corresponding face image in an adjacent
%   plot.  A click away from any point ends the interactive session.

clf;

yax = axes('position', [.1,.1,.7,.8]);
fax = axes('position', [.85,.8,.1,.1]);
axis off;

axes(yax);
hh = plot(Y(1,:), Y(2,:), '.');


disp(['=== click on a point to display the face; '...
      'click on the background to end ===']);

for ii = 1:100
  axes(yax);
  click = selectdatum('Handle', hh, 'Verbose', 0, 'MaxAttempts', 1);
  if (~isempty(click))
    axes(fax);
    showfreyface(X(:,click));
    axis off;
  else
    break;
  end
end
  

function ii = selectdatum(varargin)
% i = selectdatum(...): select datum by mouse input.
%
% SELECTDATUM waits for the user to click in the current axes and then
% returns the index of the data point within the current line object
% that lies closest to the clicked location.  
%
% OPTIONS:
% 'Handle'	[gco]	handle to line object
% 'Highlight'   ['red'] temporary object color (empty for no highlight)
% 'MaxSlop'	[10]	max distance (in points) between click and datum
% 'Verbose'	[0]	give user instructions
% 'MaxAttempts'	[5]	return empty after this many failed attempts
%
% See also: GINPUT.

% OPTIONS:
Handle = [];                % [gco] handle to line object
Highlight = 'red';          % temporary object color (empty for no highlight)
MaxSlop = 10;               % max distance (in points) between click and datum
Verbose = 0;                % give user instructions
MaxAttempts = 5;            % return empty after this many failed attempts
assignopts('ignorecase', who, varargin);

if (isempty(Handle))
  Handle = gco;
end

if (isempty(Handle))
  Handle = findobj(gca, 'type', 'line');
  if length(Handle) > 1
    Handle = Handle(1);
  end
end

if (isempty(Handle))
  error ('no plots!');
end

if isempty(strmatch(get(Handle, 'type'), {'line', 'hggroup'}))
  error('(current) object must be a line or group');
end

if ~isempty(Highlight)
  oldcolor = get(Handle, 'Color');
  set(Handle, 'Color', Highlight);
end

ii = [];

[dux,duy] = dataunits('points');
xx = get(Handle, 'xdata')/dux;
yy = get(Handle, 'ydata')/duy;

if Verbose
  if isempty(Highlight)
    disp('select a point in the current plot');
  else
    disp('select a point in the highlighted plot');
  end
end

for attempt = 1:MaxAttempts
  [x,y] = ginput(1);

  [mindist,ii] = min((x/dux - xx).^2 + (y/duy - yy).^2);
  if (mindist > MaxSlop.^2)
    ii = [];
    if (attempt < MaxAttempts)
      if Verbose disp('click was not near a data point; try again'); end
    end
  else
    break
  end
end

if (~isempty(Highlight))
  set(Handle, 'Color', oldcolor);
end  

function remain = assignopts (opts, varargin)
% assignopts - assign optional arguments (matlab 5 or higher)
%
%   REM = ASSIGNOPTS(OPTLIST, 'VAR1', VAL1, 'VAR2', VAL2, ...)
%   assigns, in the caller's workspace, the values VAL1,VAL2,... to
%   the variables that appear in the cell array OPTLIST and that match
%   the strings 'VAR1','VAR2',... .  Any VAR-VAL pairs that do not
%   match a variable in OPTLIST are returned in the cell array REM.
%   The VAR-VAL pairs can also be passed to ASSIGNOPTS in a cell
%   array: REM = ASSIGNOPTS(OPTLIST, {'VAR1', VAL1, ...});
%
%   By default ASSIGNOPTS matches option names using the strmatch
%   defaults: matches are case sensitive, but a (unique) prefix is
%   sufficient.  If a 'VAR' string is a prefix for more than one
%   option in OPTLIST, and does not match any of them exactly, no
%   assignment occurs and the VAR-VAL pair is returned in REM.
%
%   This behaviour can be modified by preceding OPTLIST with one or
%   both of the following flags:
%      'ignorecase' implies case-insensitive matches.
%      'exact'      implies exact string matches.
%   Both together imply case-insensitive, but otherwise exact, matches.
%
%   ASSIGNOPTS useful for processing optional arguments to a function.
%   Thus in a function which starts:
%		function foo(x,y,varargin)
%		z = 0;
%		assignopts({'z'}, varargin{:});
%   the variable z can be given a non-default value by calling the
%   function thus: foo(x,y,'z',4);  When used in this way, a list
%   of currently defined variables can easily be obtained using
%   WHO.  Thus if we define:
%		function foo(x,y,varargin)
%		opt1 = 1;
%               opt2 = 2;
%		rem = assignopts('ignorecase', who, varargin);
%   and call foo(x, y, 'OPT1', 10, 'opt', 20); the variable opt1
%   will have the value 10, the variable opt2 will have the
%   (default) value 2 and the list rem will have the value {'opt',
%   20}. 
% 
%   See also WARNOPTS, WHO.

ignorecase = 0;
exact = 0;

% check for flags at the beginning
while (~iscell(opts))
  switch(lower(opts))
   case 'ignorecase',
    ignorecase = 1;
   case 'exact',
    exact = 1;
   otherwise,
    error(['unrecognized flag :', opts]);
  end
  
  opts = varargin{1};
  varargin = varargin{2:end};
end

% if passed cell array instead of list, deal
if length(varargin) == 1 & iscell(varargin{1})
  varargin = varargin{1};
end

if rem(length(varargin),2)~=0,
   error('Optional arguments and values must come in pairs')
end     

done = zeros(1, length(varargin));

origopts = opts;
if ignorecase
  opts = lower(opts);
end

for i = 1:2:length(varargin)

  opt = varargin{i};
  if ignorecase
    opt = lower(opt);
  end
  
  % look for matches
  
  if exact
    match = strmatch(opt, opts, 'exact');
  else
    match = strmatch(opt, opts);
  end
  
  % if more than one matched, try for an exact match ... if this
  % fails we'll ignore this option.

  if (length(match) > 1)
    match = strmatch(opt, opts, 'exact');
  end

  % if we found a unique match, assign in the corresponding value,
  % using the *original* option name
  
  if length(match) == 1
    assignin('caller', origopts{match}, varargin{i+1});
    done(i:i+1) = 1;
  end
end

varargin(find(done)) = [];
remain = varargin;


function [dux, duy] = dataunits(units)
% [x,y] = dataunits('units'): data equivalent to physical units
%    [X,Y] = DATAUNITS('UNITS') gives the equivalent data units in the
%    current axes to the physical unit length UNITS, which must be one
%    of 'pixels', 'inches', 'centimeters', 'points' or 'normalized'.
%    This might be useful for setting lines, patches or other objects
%    without a 'units' property to a specific physical length.  But
%    see the warning below.
%
%    XY = DATAUNITS('UNITS') does the same thing, but returns a
%    two-element row-vector.
%
%    WARNING: MATLAB's conversion routines seem to be insensitive
%    to the current figure's 'paperposition'.  So the conversion
%    seems only to work reasonably for the default setting of
%    paperposition (i.e. orient portrait).

knownunits = {'pixels', 'inches', 'centimeters', 'points', 'normalized'};

if isempty(strmatch(units, knownunits))
  error(['units must be one of' sprintf(' %s', knownunits{:})]);
end

xlim = get(gca, 'xlim');
ylim = get(gca, 'ylim');

if (strmatch(units, 'normalized'))
  dux = diff(xlim);
  duy = diff(ylim);
else
  ounits = get(gca, 'units');

  set(gca, 'units', units);
  pos = get(gca, 'position');
  set(gca, 'units', ounits);
  
  dux = diff(xlim)./pos(3);
  duy = diff(ylim)./pos(4);
end

if nargout < 2
  dux = [dux, duy];
end

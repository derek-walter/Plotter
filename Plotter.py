import altair as alt
#alt.renderers.enable('notebook')
import pandas as pd
import numpy as np
import warnings

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_dtype

class Inspect(object):
    def __init__(self, df, columns=[]):
        self.columns = columns
        self.source = df.copy()
        self.x_type = None

    @staticmethod
    def is_date(item): 
        try:
            pd.to_datetime(item, infer_datetime_format=True)
            return True
        except:
            return False

    def _inspectN(self, names):
        has = set(self.source.columns).intersection(set(names))
        exists = False
        if has:
            exists = True
        has_mask = np.array([True if item in names else False for item in self.source.columns])
        return exists, has, has_mask

    def _inspectV(self, function):
        has = self.source.dtypes.apply(function)
        has_mask = has.values
        has_names = has.index[has_mask]
        return has, set(has_names), has_mask
    
    def _toDatetime(self, name):
        temp = pd.to_datetime(self.source[name], infer_datetime_format=True, utc=False)
        # Checks
        if temp.iloc[0].year == 1970:
            temp = pd.to_datetime(self.source[name], unit='ms')
        self.source[name] = temp

    '''Revalue'''
    def _melt(self):
        if not self.columns:
            _, num_names, _ = self._inspectV(is_numeric_dtype)
            names = num_names.difference({'x'})
        else:
            _, names, _ = self._inspectN(self.columns)
        self.source = self.source.melt(id_vars='x', value_vars=names)

    '''Select'''
    def _inspectSource(self):
        if self.source.shape[0] != 0:
            if self.source.shape[1] == 3:
                self._rename(values={})
                _, str_names, _ = self._inspectV(is_string_dtype)
                _, num_names, _ = self._inspectV(is_numeric_dtype)
                # melt-like
                if str_names and num_names:
                    self._rename()
                # unmelted
                if len(num_names) == 2:
                    self._melt()
                    self._rename()
            elif self.source.shape[1] == 2:
                _, num_names, _ = self._inspectV(is_numeric_dtype)
                _, date_names, _ = self._inspectV(is_datetime64_dtype)
                strs, str_names, str_mask = self._inspectV(is_string_dtype)
                str_names = strs.index[str_mask]
                date_mask = self.source.iloc[:, str_mask].apply(self.is_date, axis=0)
                if len(num_names) == 2:
                    self._rename(values={})
                    self._melt()
                    self._rename()
                elif len(num_names) == 1 and len(date_names) == 1:
                    x = date_names.pop()
                    self.rename(x, 'x')
                    self._melt()
                    self._rename()
                elif any(date_mask):
                    name = str_names[date_mask][0]
                    self.rename(name, 'x')
                    self._toDatetime('x')
                    self._melt()
                    self._rename()
                else:
                    raise NotImplementedError('Source types not valid yet.')
            elif self.source.shape[1] == 1:
                _, num_names, _ = self._inspectV(is_numeric_dtype)
                if num_names:
                    self.source.index.name = 'x'
                    self.source = self.source.reset_index()
                    self._melt()
                    self._rename()
                else:
                    temp = self.source.groupby(self.source.columns[0]).count()
                    temp.index.name = 'x'
                    self.source = temp
                    self._melt()
                    self._rename()
            else:
                self._rename(values={})
                self._melt()
                self._rename()
        else:
            raise ValueError('DF of zero length.')
        return self.source

    '''Rename'''
    def _rename(self, values={'variable', 'value'}):
        # Don't forget about x_type
        valid_out = {'x', 'variable', 'value'}
        names = set(self.source.columns)
        if valid_out not in names:
            valid_index = {'Unnamed: 0', 'X', 'Index', 'index'}
            valid_date = {'epoch_ts', 'date', 'Date', 'DATE', 'datetime', 'timestamp', 'time', 'Time', 'TIME'}
            # X
            if 'x' not in names:
                has_date_column, date_columns, _ = self._inspectN(valid_date)
                if has_date_column:
                    if len(date_columns) > 1:
                        warnings.warn('In Renaming: Multiple date columns found.')
                    name = date_columns.pop()
                    names.remove(name)
                    self.rename(name, 'x')
                    self.x_type = 'date'
                else:
                    has_index_column, index_columns, _ = self._inspectN(valid_index)
                    if has_index_column:
                        if len(index_columns) > 1:
                            warnings.warn('In Renaming: Multiple index columns found.')
                        name = index_columns.pop()
                        names.remove(name)
                        self.rename(name, 'x')
                        self.x_type = 'index'
                    else:
                        # Search for date
                        self.x_type = 'date'
                        _, date_names, date_mask = self._inspectV(is_datetime64_dtype)
                        if any(date_mask):
                            name = date_names.pop()
                            names.remove(name)
                            self.rename(name, 'x')
                        else:
                            strs, str_names, str_mask = self._inspectV(is_string_dtype)
                            str_names = strs.index[str_mask]
                            date_mask = self.source.iloc[:, str_mask].apply(self.is_date, axis=0)
                            if any(date_mask):
                                name = str_names[date_mask][0]
                                names.remove(name)
                                self.rename(name, 'x')
                                self._toDatetime('x')
                            else:
                                nums, num_names, num_mask = self._inspectV(is_numeric_dtype)
                                num_names = nums.index[num_mask]
                                date_mask = self.source.iloc[:, num_mask].apply(self.is_date, axis=0)
                                if any(date_mask):
                                    name = num_names[date_mask][0]
                                    names.remove(name)
                                    self.rename(name, 'x')
                                    self._toDatetime('x')
                                else:
                                    raise ValueError('No column is datetime convertible.')
            # Others
            if 'variable' not in names and 'variable' in values:
                _, str_names, str_mask = self._inspectV(is_string_dtype)
                str_names = str_names.difference({'x'})
                if str_names:
                    self.rename(str_names.pop(), 'variable')
                else:
                    raise ValueError('No categorical column present.')
            if 'value' not in names and 'value' in values:
                _, num_names, num_mask = self._inspectV(is_numeric_dtype)
                num_names = num_names.difference({'x', 'variable'})
                if num_names:
                    self.rename(num_names.pop(), 'value')
                else:
                    raise ValueError('No numeric column present.')
    
    def rename(self, name, new):
        self.source.rename(columns={name: new}, inplace=True)

    def all(self):
        self._inspectSource()
        return self.source, self.x_type

class Plot(object):
    '''
    Key Dictionaries:
        properties - unpacked into base of Chart().properties(**self.prop).interactive()
        xObj - " " x=alt.X(stuff, **xObj)
        xAxis - x=alt.X(axis=alt.Axis(**xAxis)) 
        xScale - x=alt.X(scale=alt.Scale(**xScale))
        yObj - y=alt.Y(**yObj)
        yAxis - y=alt.Y(axis=alt.Axis(**xAxis))
        yScale - y=alt.Y(scale=alt.Scale(**yScale))
        baseMark - alt.Chart(source, mark=alt.MarkDef(**self.baseMark))
    Key Keywords:
        Pass in force=True which skips any intelligent data check. 
        or verbose=False to silence all warnings
    Description:
        The plot generation consists of three calls:
            Init statement - Plot()
                One should put source, and properties dictionary here
            Chained methods - .base(), .legend(), or .labels() - all, some, none, any order
                Put values existing in Plot().valid_attrs in any of these methods
            Final rendering call - plot()
                Takes a save argument. See save source code if you need it. 
    Examples:
        Plot(Source).plot()
            Returns basic line chart keyed on `value` or config['y']
            If calling without config, columns must have 'value'
        Plot().info().plot()
            Returns fully toy dataset and plot.
        Plot().info().base().legend(colors='oranges').labels().plot() returns an example plot.
            Note the structure above... [Plot().info()] is what populates the toy source variable.
            Then [.base().legend(colors='oranges').labels()] are the components to be used
            Then [.plot()] is always called to render the final plot.

    '''
    def __init__(self, 
                 source=pd.DataFrame(), 
                 columns = [],
                 properties={}, 
                 xObj={},
                 xAxis={},
                 xScale={},
                 yObj={},
                 yAxis={},
                 baseMark={},
                 legendMark={},
                 labelsAxis={},
                 textMark={},
                 yScale={},
                 y2Obj={},
                 y2Axis={},
                 y2Scale={},
                 base2Mark={},
                 double=[],
                 basicLegend=None,
                 force=False, 
                 verbose=True):
        self._base=None
        self._legend=None
        self._labels=None
        self.colors=None
        self.selection=None
        self.zero=False
        self.calls = set()
        self.basicLegend=basicLegend
        self._int_attrs = {'colors'}
        self.valid_attrs = {'kind', 'scale', 'format', 'zero', 'x_label', 'y_label', 'x_format', 'y_format', 'date_label', 'timezone'}
        self.prop = {'height': 200, 'width': 600, 'title': 'Plot Title'}
        self.prop.update(properties)
        self.double = double
        self.verbose = verbose
        self.force = force
        self.columns = columns
        self.x = 'x'
        self.x_type = ':T'
        self.x_label = ''
        self.x_format = ''
        self.y = ''
        self.y_label = ''
        self._y_label = ''
        self.y_format = ''
        self._y_format = ''
        self.category_column = ''
        self.date_label = True
        self.timezone = False
        self._datebased = True
        self.source = source.copy()
        self._inspectSource()
        self.kind='line'
        self._kind='line'
        self.format = 'r'
        self.scale='linear'
        self._scale='linear'
        self._color={}
        self.xObj=xObj
        self.xAxis=xAxis
        self.xScale=xScale
        self.yObj=yObj
        self.yAxis=yAxis
        self.yScale=yScale
        self.y2Obj=y2Obj
        self.y2Axis=y2Axis
        self.y2Scale=y2Scale
        self.baseMark=baseMark
        self.legendMark=legendMark
        self.labelsAxis=labelsAxis
        self.textMark=textMark
        self.base2Mark=base2Mark
        
    def _check(self):
        if isinstance(self.scale, tuple):
            if self.double:
                if len(self.scale) == 2:
                    left, right = self.scale
                    self.scale=left
                    self._scale=right
                else:
                    raise ValueError('Scale parameter unknown')
            else:
                warnings.warn('Scale suggests double Y, but axes not mentioned. Syntax: double=[list of items], kind=(type, type), scale=(type, type)')
                if len(self.scale) >= 1:
                    self.scale = self.scale[0]
        if isinstance(self.kind, tuple):
            if self.double:
                if len(self.kind) == 2:
                    left, right = self.kind
                    self.kind=left
                    self._kind=right
                else:
                    raise ValueError('Kind parameter unknown')
            else:
                warnings.warn('Kind suggests double Y, but axes not mentioned. Syntax: double=[list of items], kind=(type, type), scale=(type, type)')
                if len(self.kind) >= 1:
                    self.kind = self.kind[0]
        if isinstance(self.y_label, tuple):
            if self.double:
                if len(self.y_label) == 2:
                    left, right = self.y_label
                    self.y_label=left
                    self._y_label=right
                else:
                    raise ValueError('Label parameter unknown')
            else:
                warnings.warn('Y Label suggests double Y, but axes not mentioned. Syntax: double=[list of items], y_label=(type, type)')
                if len(self.y_label) >= 1:
                    self.y_label = self.y_label[0]
        if isinstance(self.y_format, tuple):
            if self.double:
                if len(self.y_format) == 2:
                    left, right = self.y_format
                    self.y_format=left
                    self._y_format=right
                else:
                    raise ValueError('Label parameter unknown')
            else:
                warnings.warn('Y Label suggests double Y, but axes not mentioned. Syntax: double=[list of items], y_label=(type, type)')
                if len(self.y_format) >= 1:
                    self.y_format = self.y_format[0]
        # Any explicit properties called from self should be added here.
        self.xObj.update({
            
        })
        self.xAxis.update({
            'labelAngle':-25
        })
        self.xScale.update({
            
        })
        self.yObj.update({
            
        })
        self.yAxis.update({
            
        })
        self.yScale.update({
            
        })
        self.baseMark.update({
            'clip':True
        })
        self.legendMark.update({
            'size':250, 
            'shape':'circle', 
            'filled':True
        })
        self.textMark.update({
            'baseline':'middle',
            'color':'black',
            'size':14
        })
        self.labelsAxis.update({
            'orient':'top',
            'labelAngle':0,
            'ticks':False
        })
        self.y2Obj.update({
            
        })
        self.y2Axis.update({
            
        })
        self.y2Scale.update({
            
        })
        self.base2Mark.update({
            'clip':True
        })
        
    def _queries(self, items, name='variable'):
        _query = lambda xs: ' | '.join(['datum.{0} == "{1}"'.format(name, x) for x in xs])
        wanted = set(items)
        have = set(self.source[name].unique())
        if len(wanted.intersection(have)) > 0:
            other = have.difference(set(items))
            if wanted and other:
                return _query(other), _query(wanted)
            else:
                raise ValueError('Not enough unique values for double Y.')
        else:
            raise ValueError('Double values not in variable column.')

    def _inspectSource(self):
        if self.source.empty:
            Source = pd.DataFrame([{'date':"2019-04-04", 'variable':'TXN Vol', 'value':40000},
                                   {'date':"2019-04-05", 'variable':'TXN Vol', 'value':51000},
                                   {'date':"2019-04-06", 'variable':'TXN Vol', 'value':30000},
                                   {'date':"2019-04-04", 'variable':'Price', 'value':8502},
                                   {'date':"2019-04-05", 'variable':'Price', 'value':7195},
                                   {'date':"2019-04-06", 'variable':'Price', 'value':8295}])
            self.source = Source
            self.x='x'
        if not self.force:
            self.source, x_type = Inspect(self.source, self.columns).all()
            if x_type == 'index':
                self.x_type = ':Q'
                self._datebased = False
        else:
            if len(self.source.columns) != 3:
                raise NotImplementedError('Please use a melted DF if forcing.')

    def _parseArgs(self, call, **kwargs):
        self.calls.add(call)
        colors = kwargs.get('colors')
        if colors:
            if isinstance(colors, str):
                self.colors = {'scheme':colors}
            elif isinstance(colors, list):
                self.colors = {'range':colors}
        if not self.colors:
            self.colors = {'scheme':'blues'}
        for k, v in kwargs.items():
            if k in self.valid_attrs:
                setattr(self, k, v)
            else:
                if k not in self._int_attrs:
                    warnings.warn(f'Keyword argument {k} not yet supported. Use dict assignment.')

    def _addColor(self):
        # Note, color is a dict, selector is an Altair object
        if not self.selection:
            selection = alt.selection_multi(fields=['variable'])
            color = alt.condition(selection,
                              alt.Color('variable:N', scale=alt.Scale(**self.colors), legend=None),
                              alt.value('#f2f2f2'))
            self.selection=selection
            self._color={'color':color}
        
    def info(self):
        print(self.source)
        print(self.prop)
        self.base()
        return self

    def plot(self, save=False, filepath='', **kwargs):
        self._check()
        if 'base' not in self.calls:
            self._parseArgs(call='base', **kwargs)
        if 'legend' in self.calls:
            self.legend(internal=True)
        else:
            self._color = {'color':alt.Color('variable', scale=alt.Scale(**self.colors), legend=self.basicLegend)}
        if 'labels' in self.calls:
            self.labels(internal=True)
        if not self._base:
            self.base(internal=True)
        if 'legend' in self.calls:
            self._base = alt.hconcat(self._base, self._legend)
        if 'labels' in self.calls:
            self._base = alt.vconcat(self._labels, self._base)
        if not save:
            return self._base
        else:
            if filepath:
                items = filepath.split('/')[-1].split('.')
                if set(items).intersection({'png', 'html', 'jpeg', 'svg'}):
                    self._base.save(filepath, webdriver='firefox', **kwargs)
                else:
                    raise ValueError('Filepath doesn\'t contain a recognized filetype ending.')
            else:
                raise ValueError('Filepath empty, are you trying to save? Extra args are passed into altair\'s save method.')


    def labels(self, internal=False, **kwargs):
        self._parseArgs(call='labels', **kwargs)
        if internal:
            temp = self.source.groupby('variable').last().reset_index()
            if self.date_label:
                if type(self.date_label) == bool:
                    if self._datebased:
                        max_time = temp['x'].max()
                        if self.timezone:
                            formatted_time = pd.to_datetime(max_time, infer_datetime_format=True, utc=True)
                            _formatted_time = str(formatted_time.tz_localize(None)) + ' ' + str(formatted_time.tzinfo)
                        else:
                            formatted_time = pd.to_datetime(max_time, infer_datetime_format=True)
                            _formatted_time = str(formatted_time.tz_localize(None))
                        temp_time = pd.DataFrame([{'variable':'Time', 'x':max_time, 'value':_formatted_time}])
                elif type(self.date_label) == str:
                    if self.date_label in {'Date', 'date', 'days', 'day'}:
                        max_time = temp['x'].max()
                        formatted_time = pd.to_datetime(max_time, infer_datetime_format=True).date()
                        temp_time = pd.DataFrame([{'variable':'Time', 'x':max_time, 'value':str(formatted_time)}])
                    elif self.date_label in {'Time', 'time', 'hour', 'hours'}:
                        max_time = temp['x'].max()
                        formatted_time = pd.to_datetime(max_time, infer_datetime_format=True).time()
                        temp_time = pd.DataFrame([{'variable':'Time', 'x':max_time, 'value':str(formatted_time)}])
                    else:
                        temp_time = pd.DataFrame([{'variable':'  ', 'x':str(self.date_label), 'value':str(self.date_label)}])
                else:
                    raise ValueError('Unsupported date_label argument.')
                _width = 80 + 4*int(len(temp_time['value'].values[0]))
                width = self.prop.get('width')
                width1, width2 = width - _width, _width
                labels = alt.Chart(temp).mark_text(**self.textMark).encode(
                        x=alt.X('variable:O',
                                axis=alt.Axis(**self.labelsAxis),
                                title=None),
                        text=alt.Text('value:Q',
                                      format=self.format)).properties(width=width1,
                                                    height=30,
                                                    title='')

                time_label = alt.Chart(temp_time).mark_text(**self.textMark).encode(
                        x=alt.X('variable:O',
                                axis=alt.Axis(**self.labelsAxis),
                                title=None),
                        text=alt.Text('value:O',
                                     )).properties(width=width2,
                                                             height=30,
                                                             title='')
                labels = alt.hconcat(labels, time_label, spacing=0, title=alt.TitleParams(text=self.prop.get('title', 'Title Needed'), anchor='middle'))
            else:
                temp = self.source.groupby('variable').last().reset_index()
                labels = alt.Chart(temp).mark_text(**self.textMark).encode(
                        x=alt.X('variable:O',
                                axis=alt.Axis(**self.labelsAxis),
                                title=None),
                        text=alt.Text('value:Q',
                                      format=self.format)).properties(width=self.prop.get('width'),
                                                              height=30,
                                                              title=self.prop.get('title', 'Title Needed'))
            if self.prop.get('title'):
                self.prop.pop('title')
            self._labels = labels
        return self

    def legend(self, internal=False, **kwargs):
        self._parseArgs(call='legend', **kwargs)
        if internal:
            self._addColor()
            legend = alt.Chart(self.source).mark_point(**self.legendMark).encode(
                    y=alt.Y(f'variable:N',
                            axis=alt.Axis(orient='right',
                                          grid=False),
                            title=''),
                    **self._color
            ).properties(
                width=30,
                height=self.prop.get('height'),
            ).add_selection(self.selection)
            self._legend = legend
        return self

    def base(self, internal=False, **kwargs):
        self._parseArgs(call='base', **kwargs)
        if internal:
            if self.double:
                return self._double()
            else:
                base = alt.Chart(self.source, mark=alt.MarkDef(self.kind,
                                                               **self.baseMark)).encode(
                        x=alt.X(f'{self.x + self.x_type}',
                                title=self.x_label,
                                scale=alt.Scale(**self.xScale),
                                axis=alt.Axis(format=self.x_format,
                                              **self.xAxis),
                                **self.xObj),
                        y=alt.Y(f'value:Q',
                                title=self.y_label,
                                scale=alt.Scale(type=self.scale,
                                                **self.yScale),
                                axis=alt.Axis(format=self.y_format, **self.yAxis),
                                **self.yObj),
                        **self._color,
                ).properties(**self.prop).interactive()
                self._base = base
        return self
    
    def _double(self):     
        if isinstance(self.double, str):
            first, other = self._queries([self.double])
        elif isinstance(self.double, list) or isinstance(self.double, set):
            first, other = self._queries(self.double)
        else:
            raise ValueError('Double arguments not understood.')
        chart_one = alt.Chart(self.source, mark=alt.MarkDef(self.kind,
                                                     **self.baseMark)).encode(
            x=alt.X(f'{self.x + self.x_type}',
                        title=self.x_label,
                        scale=alt.Scale(**self.xScale),
                        axis=alt.Axis(format=self.x_format,
                                      **self.xAxis),
                        **self.xObj),
            y=alt.Y(f'value:Q',
                        title=self.y_label,
                        scale=alt.Scale(type=self.scale,
                                        **self.yScale),
                        axis=alt.Axis(format=self.y_format, **self.yAxis),
                        **self.yObj),
            **self._color,
        ).transform_filter(first)

        chart_two = alt.Chart(self.source, mark=alt.MarkDef(self._kind,
                                                            **self.base2Mark)).encode(
            x=alt.X(f'{self.x + self.x_type}',
                        title=self.x_label,
                        scale=alt.Scale(**self.xScale),
                        axis=alt.Axis(format=self.x_format,
                                      **self.xAxis),
                        **self.xObj),
            y=alt.Y(f'value:Q',
                        title=self._y_label,
                        scale=alt.Scale(type=self._scale,
                                        **self.y2Scale),
                        axis=alt.Axis(format=self._y_format, **self.y2Axis),
                        **self.y2Obj),
            **self._color,
        ).transform_filter(other)

        self._base = alt.layer(chart_one, chart_two).properties(**self.prop).resolve_scale(y='independent').interactive()
        return self

class OriginalPlot(object):
    '''
    Key Dictionaries:
        properties - unpacked into base of Chart().properties(**self.prop).interactive()
        xObj - " " x=alt.X(stuff, **xObj)
        xAxis - x=alt.X(axis=alt.Axis(**xAxis)) 
        xScale - x=alt.X(scale=alt.Scale(**xScale))
        yObj - y=alt.Y(**yObj)
        yAxis - y=alt.Y(axis=alt.Axis(**xAxis))
        yScale - y=alt.Y(scale=alt.Scale(**yScale))
        baseMark - alt.Chart(source, mark=alt.MarkDef(**self.baseMark))
    Key Keywords:
        Pass in force=True which skips any intelligent data check. 
        or verbose=False to silence all warnings
    Description:
        The plot generation consists of three calls:
            Init statement - Plot()
                One should put source, and properties dictionary here
            Chained methods - .base(), .legend(), or .labels() - all, some, none, any order
                Put values existing in Plot().valid_attrs in any of these methods
            Final rendering call - plot()
                Takes a save argument. See save source code if you need it. 
    Examples:
        Plot(Source).plot()
            Returns basic line chart keyed on `value` or config['y']
            If calling without config, columns must have 'value'
        Plot().info().plot()
            Returns fully toy dataset and plot.
        Plot().info().base().legend(colors='oranges').labels().plot() returns an example plot.
            Note the structure above... [Plot().info()] is what populates the toy source variable.
            Then [.base().legend(colors='oranges').labels()] are the components to be used
            Then [.plot()] is always called to render the final plot.

    '''
    def __init__(self, 
                 source=pd.DataFrame(), 
                 columns = [],
                 properties={}, 
                 xObj={},
                 xAxis={},
                 xScale={},
                 yObj={},
                 yAxis={},
                 baseMark={},
                 yScale={},
                 y2Obj={},
                 y2Axis={},
                 y2Scale={},
                 base2Mark={},
                 double=[],
                 basicLegend=None,
                 force=False, 
                 verbose=True):
        self._base=None
        self._legend=None
        self._labels=None
        self.colors=None
        self.selection=None
        self.zero=False
        self.calls = set()
        self.basicLegend=basicLegend
        self._int_attrs = {'colors'}
        self.valid_attrs = {'kind', 'scale', 'format', 'zero', 'x_label', 'y_label', 'x_format', 'y_format'}
        self.prop = {'height': 200, 'width': 400, 'title': 'Plot Title'}
        self.prop.update(properties)
        self.double = double
        self.verbose = verbose
        self.force = force
        self.columns = columns
        self.x = 'x'
        self.x_type = ':T'
        self.x_label = ''
        self.x_format = ''
        self.y = ''
        self.y_label = ''
        self._y_label = ''
        self.y_format = ''
        self._y_format = ''
        self.category_column = ''
        self.source = source.copy()
        self._inspectSource()
        self.kind='line'
        self._kind='line'
        self.format = 'r'
        self.scale='linear'
        self._scale='linear'
        self._color={}
        self.xObj=xObj
        self.xAxis=xAxis
        self.xScale=xScale
        self.yObj=yObj
        self.yAxis=yAxis
        self.yScale=yScale
        self.y2Obj=y2Obj
        self.y2Axis=y2Axis
        self.y2Scale=y2Scale
        self.baseMark=baseMark
        self.base2Mark=base2Mark
        
    def _check(self):
        if isinstance(self.scale, tuple):
            if self.double:
                if len(self.scale) == 2:
                    left, right = self.scale
                    self.scale=left
                    self._scale=right
                else:
                    raise ValueError('Scale parameter unknown')
            else:
                warnings.warn('Scale suggests double Y, but axes not mentioned. Syntax: double=[list of items], kind=(type, type), scale=(type, type)')
                if len(self.scale) >= 1:
                    self.scale = self.scale[0]
        if isinstance(self.kind, tuple):
            if self.double:
                if len(self.kind) == 2:
                    left, right = self.kind
                    self.kind=left
                    self._kind=right
                else:
                    raise ValueError('Kind parameter unknown')
            else:
                warnings.warn('Kind suggests double Y, but axes not mentioned. Syntax: double=[list of items], kind=(type, type), scale=(type, type)')
                if len(self.kind) >= 1:
                    self.kind = self.kind[0]
        if isinstance(self.y_label, tuple):
            if self.double:
                if len(self.y_label) == 2:
                    left, right = self.y_label
                    self.y_label=left
                    self._y_label=right
                else:
                    raise ValueError('Label parameter unknown')
            else:
                warnings.warn('Y Label suggests double Y, but axes not mentioned. Syntax: double=[list of items], y_label=(type, type)')
                if len(self.y_label) >= 1:
                    self.y_label = self.y_label[0]
        if isinstance(self.y_format, tuple):
            if self.double:
                if len(self.y_format) == 2:
                    left, right = self.y_format
                    self.y_format=left
                    self._y_format=right
                else:
                    raise ValueError('Label parameter unknown')
            else:
                warnings.warn('Y Label suggests double Y, but axes not mentioned. Syntax: double=[list of items], y_label=(type, type)')
                if len(self.y_format) >= 1:
                    self.y_format = self.y_format[0]
        #if self.kind=='bar' and self.zero==False:
        #    self.zero=True
        #if self.kind=='bar' and self.scale=='log':
        #    if self.force:
        #        self.yObj.update({'stack':False})
        #    if self.verbose:
        #        warnings.warn('Unsafe scale for zero valued data')
        # Any explicit properties called from self should be added here.
        self.xObj.update({
            
        })
        self.xAxis.update({
            
        })
        self.xScale.update({
            
        })
        self.yObj.update({
            
        })
        self.yAxis.update({
            
        })
        self.yScale.update({
            
        })
        self.baseMark.update({
            
        })
        self.y2Obj.update({
            
        })
        self.y2Axis.update({
            
        })
        self.y2Scale.update({
            
        })
        self.base2Mark.update({
            
        })
        
    def _queries(self, items, name='variable'):
        _query = lambda xs: ' | '.join(['datum.{0} == "{1}"'.format(name, x) for x in xs])
        wanted = set(items)
        have = set(self.source[name].unique())
        if len(wanted.intersection(have)) > 0:
            other = have.difference(set(items))
            if wanted and other:
                return _query(other), _query(wanted)
            else:
                raise ValueError('Not enough unique values for double Y.')
        else:
            raise ValueError('Double values not in variable column.')

    def _inspectSource(self):
        if self.source.empty:
            Source = pd.DataFrame([{'date':"2019-04-04", 'variable':'TXN Vol', 'value':40000},
                                   {'date':"2019-04-05", 'variable':'TXN Vol', 'value':51000},
                                   {'date':"2019-04-06", 'variable':'TXN Vol', 'value':30000},
                                   {'date':"2019-04-04", 'variable':'Price', 'value':8502},
                                   {'date':"2019-04-05", 'variable':'Price', 'value':7195},
                                   {'date':"2019-04-06", 'variable':'Price', 'value':8295}])
            self.source = Source
            self.x='x'
        if not self.force:
            self.source, x_type = Inspect(self.source, self.columns).all()
            if x_type == 'index':
                self.x_type = ':Q'
        else:
            if len(self.source.columns) != 3:
                raise NotImplementedError('Please use a melted DF if forcing.')

    def _parseArgs(self, call, **kwargs):
        self.calls.add(call)
        colors = kwargs.get('colors')
        if colors:
            if isinstance(colors, str):
                self.colors = {'scheme':colors}
            elif isinstance(colors, list):
                self.colors = {'range':colors}
        if not self.colors:
            self.colors = {'scheme':'blues'}
        for k, v in kwargs.items():
            if k in self.valid_attrs:
                setattr(self, k, v)
            else:
                if k not in self._int_attrs:
                    warnings.warn(f'Keyword argument {k} not yet supported. Use dict assignment.')

    def _addColor(self):
        # Note, color is a dict, selector is an Altair object
        if not self.selection:
            selection = alt.selection_multi(fields=['variable'])
            color = alt.condition(selection,
                              alt.Color('variable:N', scale=alt.Scale(**self.colors), legend=None),
                              alt.value('#f2f2f2'))
            self.selection=selection
            self._color={'color':color}
        
    def info(self):
        print(self.source)
        print(self.prop)
        self.base()
        return self

    def plot(self, save=False, filepath='', **kwargs):
        self._check()
        if 'base' not in self.calls:
            self._parseArgs(call='base', **kwargs)
        if 'legend' in self.calls:
            self.legend(internal=True)
        else:
            self._color = {'color':alt.Color('variable', scale=alt.Scale(**self.colors), legend=self.basicLegend)}
        if 'labels' in self.calls:
            self.labels(internal=True)
        if not self._base:
            self.base(internal=True)
        if 'legend' in self.calls:
            self._base = alt.hconcat(self._base, self._legend)
        if 'labels' in self.calls:
            self._base = alt.vconcat(self._labels, self._base)
        if not save:
            return self._base
        else:
            if filepath:
                items = filepath.split('/')[-1].split('.')
                if set(items).intersection({'png', 'html', 'jpeg', 'svg'}):
                    self._base.save(filepath, webdriver='firefox', **kwargs)
                else:
                    raise ValueError('Filepath doesn\'t contain a recognized filetype ending.')
            else:
                raise ValueError('Filepath empty, are you trying to save? Extra args are passed into altair\'s save method.')


    def labels(self, internal=False, **kwargs):
        self._parseArgs(call='labels', **kwargs)
        if internal:
            temp = self.source.groupby('variable').last().reset_index()
            labels = alt.Chart(temp).mark_text(baseline='middle',
                                               color='black',
                                               size=14).encode(
                    x=alt.X('variable:O',
                            axis=alt.Axis(orient='top',
                                          labelAngle=0,
                                          ticks=False),
                            title=None),
                    text=alt.Text(f'value:Q',
                                  format=self.format)).properties(width=self.prop.get('width'),
                                                          height=30,
                                                          title=self.prop.get('title', 'Title Needed'))
            if self.prop.get('title'):
                self.prop.pop('title')
            self._labels = labels
        return self

    def legend(self, internal=False, **kwargs):
        self._parseArgs(call='legend', **kwargs)
        if internal:
            self._addColor()
            legend = alt.Chart(self.source).mark_point(size=250, shape='square').encode(
                    y=alt.Y(f'variable:N',
                            axis=alt.Axis(orient='right',
                                          grid=False),
                            title=''),
                    **self._color
            ).properties(
                width=30,
                height=self.prop.get('height'),
            ).add_selection(self.selection)
            self._legend = legend
        return self

    def base(self, internal=False, **kwargs):
        self._parseArgs(call='base', **kwargs)
        if internal:
            if self.double:
                return self._double()
            else:
                base = alt.Chart(self.source, mark=alt.MarkDef(self.kind, 
                                                               clip=True, 
                                                               **self.baseMark)).encode(
                        x=alt.X(f'{self.x + self.x_type}',
                                title=self.x_label,
                                scale=alt.Scale(**self.xScale),
                                axis=alt.Axis(format=self.x_format, 
                                              labelAngle=-25,
                                              **self.xAxis),
                                **self.xObj),
                        y=alt.Y(f'value:Q',
                                title=self.y_label,
                                scale=alt.Scale(type=self.scale,
                                                **self.yScale),
                                axis=alt.Axis(format=self.y_format, **self.yAxis),
                                **self.yObj),
                        **self._color,
                ).properties(**self.prop).interactive()
                self._base = base
        return self
    
    def _double(self):     
        if isinstance(self.double, str):
            first, other = self._queries([self.double])
        elif isinstance(self.double, list) or isinstance(self.double, set):
            first, other = self._queries(self.double)
        else:
            raise ValueError('Double arguments not understood.')
        chart_one = alt.Chart(self.source, mark=alt.MarkDef(self.kind,
                                                     clip=True,
                                                     **self.baseMark)).encode(
            x=alt.X(f'{self.x + self.x_type}',
                        title=self.x_label,
                        scale=alt.Scale(**self.xScale),
                        axis=alt.Axis(format=self.x_format, 
                                      labelAngle=-25,
                                      **self.xAxis),
                        **self.xObj),
            y=alt.Y(f'value:Q',
                        title=self.y_label,
                        scale=alt.Scale(type=self.scale,
                                        **self.yScale),
                        axis=alt.Axis(format=self.y_format, **self.yAxis),
                        **self.yObj),
            **self._color,
        ).transform_filter(first)

        chart_two = alt.Chart(self.source, mark=alt.MarkDef(self._kind,
                                                            clip=True,
                                                            **self.base2Mark)).encode(
            x=alt.X(f'{self.x + self.x_type}',
                        title=self.x_label,
                        scale=alt.Scale(**self.xScale),
                        axis=alt.Axis(format=self.x_format, 
                                      labelAngle=-25,
                                      **self.xAxis),
                        **self.xObj),
            y=alt.Y(f'value:Q',
                        title=self._y_label,
                        scale=alt.Scale(type=self._scale,
                                        **self.y2Scale),
                        axis=alt.Axis(format=self._y_format, **self.y2Axis),
                        **self.y2Obj),
            **self._color,
        ).transform_filter(other)

        self._base = alt.layer(chart_one, chart_two).properties(**self.prop).resolve_scale(y='independent').interactive()
        return self
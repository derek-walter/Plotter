import altair as alt
alt.renderers.enable('notebook')
import pandas as pd
import numpy as np
class Plot(object):
    '''
    Description:
        The plot generation consists of three calls:
            Init statement - Plot()
                One should put source, and properties dictionary here
            Chained methods - .base(), .legend(), or .labels() - all, some, none, any order
                Put values existing in Plot().valid_attrs in any of these methods
            Final rendering call - plot()
                Will eventually take a save argument.
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
    def __init__(self, source=pd.DataFrame(), properties={}):
        self._base=None
        self._legend=None
        self._labels=None
        self.valid_attrs = {'kind', 'scale', 'format', 'zero', 'x_label', 'y_label', 'x_format', 'y_format'}
        self.prop={'height': 200, 'width': 400, 'title': 'Plot Title'}
        self.prop.update(properties)
        self.x = 'date'
        self.x_type = ':T'
        self.x_label = ''
        self.x_format = ''
        self.y = ''
        self.y_label = ''
        self.y_format = ''
        self.category_column = ''
        self.source = source
        self._inspectSource(self.source)
        self.other={}
        self._color={}
        self.colors=None
        self.kind='line'
        self.format = 'r'
        self.scale='linear'
        self.zero=False

    def _inspectSource(self, df):
        if df.empty:
            Source = pd.DataFrame([{'time':"2019-04-04", 'variable':'TXN Vol', 'value':40000},
                                   {'time':"2019-04-05", 'variable':'TXN Vol', 'value':51000},
                                   {'time':"2019-04-06", 'variable':'TXN Vol', 'value':30000},
                                   {'time':"2019-04-04", 'variable':'Price', 'value':8502},
                                   {'time':"2019-04-05", 'variable':'Price', 'value':7195},
                                   {'time':"2019-04-06", 'variable':'Price', 'value':8295}])
            self.source = Source
            self.x='time'
        elif len(df.columns) == 3:
            self.source = self._meltedCheck(df)
        else:
            raise NotImplementedError('Please use a melted DF.')

    def _parseArgs(self, call, **kwargs):
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
        if self.kind=='bar' and self.zero==False:
            self.zero=True
        if self.kind=='bar' and self.scale!='linear':
            self.other = {'stack':False}

    def _meltedCheck(self, source, recall=False):
        def dtype_check(source, dtype):
            if source.shape[1] != 0:
                return source.dtypes.apply(lambda x: np.issubdtype(x, dtype)).values
            else:
                return [False]

        def value_check(source):
            date_check = dtype_check(source, np.datetime64)
            if any(date_check):
                nex = source.iloc[:, ~date_check]
                str_check = dtype_check(nex, np.object_)
                if any(str_check):
                    nex = nex.iloc[:, ~str_check]
                    val_check = dtype_check(nex, np.number)
                    if any(val_check):
                        return True

        def name_check(source):
            if set(source.columns) == {'date', 'variable', 'value'}:
                return True
            else:
                return False

        def rename_x(source):
            valid = {'date', 'Date', 'time', 'Time', 'datetime', 'epoch_ts', 'Index'}
            target = set(source.columns).intersection(valid)
            if target:
                if len(target) > 1:
                    raise ValueError('Multiple x\'s found.')
                else:
                    name = target.pop()
                    if name == 'Index':
                        source.rename(columns={'Index':'date'}, inplace=True)
                        self.x_type = ':Q'
                    else:
                        source.rename(columns={name:'date'}, inplace=True)

        def is_date(item):
            try:
                pd.to_datetime(item)
                return True
            except:
                return False

        def find_date(source):
            mask = dtype_check(source, np.object_)
            if any(mask):
                return source.iloc[:, mask].apply(is_date, axis=0)
            else:
                return pd.Series([False])

        def revalue_x(source):
            results = find_date(source)
            values = results.values
            if any(values):
                name = results.index[values].values[0]
                source[name] = pd.to_datetime(source[name])
                if source[name].iloc[0].year == 1970:
                    source[name] = pd.to_datetime(source[name], units='ms')
                source.rename(columns={name:'date'}, inplace=True)
            else:
                raise ValueError('No column is datetime-convertible')

        def revalue(source):
            def find_object(source, ob):
                mask = dtype_check(source, ob)
                return source.iloc[:, mask].columns[0]
            name = find_object(source, np.object_)
            source.rename(columns={name:'variable'}, inplace=True)
            name = find_object(source, np.number)
            source.rename(columns={name:'value'}, inplace=True)

        # Main
        if value_check(source):
            if name_check(source):
                return source
            else:
                if len(set(source.columns).union({'variable', 'value'})) == 2:
                    rename_x(source)
                    if name_check(source):
                        return source
                    else:
                        raise ValueError('X value not understood.')
                else:
                    revalue(source)
                    return self._meltedCheck(source, recall=True)
        else:
            revalue_x(source)
            if not recall:
                return self._meltedCheck(source, recall=True)
            else:
                raise ValueError('X value not convertible')


    def info(self):
        print(self.source)
        print(self.prop)
        self.base()
        return self

    def plot(self, save=False, filepath='', **kwargs):
        if not self._base:
            self.base()
        if self._legend:
            self._base = alt.hconcat(self._base, self._legend)
        if self._labels:
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


    def _addColor(self):
        # Note, color is a dict, selector is an Altair object
        selection = alt.selection_multi(fields=['variable'])
        color = alt.condition(selection,
                          alt.Color('variable:N', scale=alt.Scale(**self.colors), legend=None),
                          alt.value('#f2f2f2'))
        self._color={'color':color}
        self.selection=selection

    def labels(self, **kwargs):
        self._parseArgs(call='labels', **kwargs)

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
        self.base()
        return self

    def legend(self, **kwargs):
        self._parseArgs(call='legend', **kwargs)

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
        self.base()
        return self

    def base(self, **kwargs):
        self._parseArgs(call='base', **kwargs)
        """
        params:
            kind: tested on line and bar
            scale_type: linear, log...
        """
        base = alt.Chart(self.source, mark=alt.MarkDef(self.kind, clip=True)).encode(
                x=alt.X(f'{self.x + self.x_type}',
                        title=self.x_label,
                        axis=alt.Axis(format=self.x_format,
                                      labelAngle=-25)),
                y=alt.Y(f'value:Q',
                        title=self.y_label,
                        scale=alt.Scale(type=self.scale),
                        **self.other),
                **self._color,
        ).properties(**self.prop).interactive()

        self._base = base
        return self

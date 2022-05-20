import dash


# external CSS stylesheets
external_stylesheets = ['assets/base.css', 'clinical-analytics.css']

external_scripts = ['resizing.js']

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_scripts = external_scripts,
    external_stylesheets = external_stylesheets
)

app.title = '양산'

server = app.server
app.config.suppress_callback_exceptions = True


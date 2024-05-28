import base64
import datetime
import io
import plotly.graph_objs as go
import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table
from dash import callback_context
from dash.exceptions import PreventUpdate
import pandas as pd 
import plotly.express as px
from sklearn.preprocessing import RobustScaler
import pandas as pd #Datenimport
import numpy as np #numerische Operationen #Plotten
from sklearn.model_selection import train_test_split # Split zwischen Trainings-und Testset
from sklearn.svm import SVC # SVC = support vector classifier 
from sklearn import svm # svm = support vector machine


# In[ ]:


from dash import html
import dash_loading_spinners as dls
external_stylesheets =['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
colors = {
    "graphBackground": "#ffffff",
    "background": "#ffffff",
    "text": "#000000"
}
app.layout = html.Div(
     children=[
             html.Div(
                 id="div-loading",
                 children=[
                     dls.Hash(
                         fullscreen=True, 
                         id="loading-whole-app"
                     )
                 ]
             ),
             html.Div(
                 className="div-app",
                 id="div-app",
                 children = [ 
                     html.Div([
                         html.H3('Welcome to Blanc de Noir check'),
                         html.H5('Please insert the CIE L*a*b* parameters in the boxes below'),
                         html.Div([
                             dcc.Input(id='L*',
                                       type='number',
                                       placeholder="Insert L*-coordinate",
                                       style={"margin-right":'15px'}),  
                             dcc.Input(id='a*',
                                       type='number',
                                       placeholder="Insert a*-coordinate",
                                       style={"margin-right":'15px'}),           
                             dcc.Input(id='b*',
                                       type='number',
                                       placeholder='Insert b*-coordinate',
                                       style={"margin-right":'15px'}),
                             html.Button('Submit',id='sub',n_clicks = 0),
                            ]),
                        ]), 
                        html.Div(id='Output_SVM'),
                        html.Div([
                            html.H3('Statistical Distribution')
                        ]),
                        html.Div([
                        html.Div([
                            dls.Hash(dcc.Graph(id="boxl", config=dict({'editable':True,'edits':dict({'annotationPosition':True})})))],className="four columns"),
                        html.Div([
                            dls.Hash(dcc.Graph(id="boxa", config=dict({'editable':True,'edits':dict({'annotationPosition':True})})))],className="four columns"),
                        html.Div([
                            dls.Hash(dcc.Graph(id="boxb", config=dict({'editable':True,'edits':dict({'annotationPosition':True})})))],className="four columns"),      
                    ]),
                        html.Div([
                            html.H3('Two-dimensional projection')
                        ]),

                        html.Div([
                            html.Div([
                                dls.Hash(dcc.Graph(id="Laplot", config=dict({'editable':True,'edits':dict({'annotationPosition':True})})))],className="four columns"),
                            html.Div([
                                dls.Hash(dcc.Graph(id="Lbplot", config=dict({'editable':True,'edits':dict({'annotationPosition':True})})))],className="four columns"),
                            html.Div([
                                dls.Hash(dcc.Graph(id="abplot", config=dict({'editable':True,'edits':dict({'annotationPosition':True})})))],className="four columns")      
                        ]),




                        html.Div([
                            html.H3('Three-dimensional projection'),
                            html.Div([dls.Hash(dcc.Graph(id='3d',config=dict({'editable':True,'edits':dict({'annotationPosition':True})}))
                                               )])],className='six columns'),

                                    ]
                                 )
                             ]
                         )
@app.callback(
    Output("div-loading", "children"),
    [
        Input("div-app", "loading_state")
    ],
    [
        State("div-loading", "children"),
    ]
)

def hide_loading_after_startup(
    loading_state, 
    children
    ):
    if children:
        print("remove loading spinner!")
        return None
    print("spinner already gone!")
    raise PreventUpdate
    
@app.callback(Output('Output_SVM','children'),
             [
                 State('L*','value'),
                 State('a*','value'),
                 State('b*','value'),
                 Input('sub','n_clicks')
             ])
def display_Click(L_value,a_value,b_value,Submit):
    changed_id=[p['prop_id'] for p in callback_context.triggered][0]
    if 'sub' in changed_id: 
        try:  
            url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
            df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')
            #data curation
            y_id=df['target_id']
            y=df['target']
            X_data = df.drop(['target','target_id'], axis=1) 
            #scaling the data
            scaler = RobustScaler() 
            X_scale = scaler.fit_transform(X_data)
            X_scale_plot=pd.DataFrame(X_scale, columns=['L*','a*','b*'])
            # prepating the plot   
            X= X_scale_plot[['L*','a*','b*']]
            # add new data from Input boxes
            # Computing SVM model with three variables 
            model =svm.SVC(kernel='rbf', gamma=0.8, C=21,) 
            clf = model.fit(X, y)
            q1L,q3L=np.percentile(df['L*'],[25,75]) #fetch 25% and 75% quantils for l*
            IQRL=q3L-q1L # calculate Interquantilerange for L*
            scaled_new_L= (L_value-df['L*'].median(axis=0))/IQRL # Scaling the new data without changing the model
            q1a,q3a=np.percentile(df['a*'],[25,75]) ##fetch 25% and 75% quantils for *
            IQRa=q3a-q1a  # calculate Interquantilerange for a*
            scaled_new_a=(a_value-df['a*'].median(axis=0))/IQRa # Scaling the new data without changing the model
            q1b,q3b = np.percentile(df['b*'],[25,75]) #fetch 25% and 75% quantils for b*
            IQRb = q3b-q1b # calculate Interquantilerange for b*
            scaled_new_b=(b_value-df['b*'].median(axis=0))/IQRb # Scaling the new data without changing the model
            new_data=np.array([scaled_new_L,scaled_new_a,scaled_new_b]).reshape(1,-1)
            pred_new_wine=clf.predict(new_data)
            if pred_new_wine == 1:
                return f'this wine is most liekely a Blanc de Noir.' 
            else:
                return f'this wine is most likely NOT a Blanc de Noir.'
        except:
            return f'Error 404: CIE L*a*b* coordinates not found or readable. Please insert your CIE L*a*b* coordinates'

    else:
        return None


        
@app.callback(Output('boxl', 'figure'),
              [ 
                  State('L*','value'),
                  Input('sub','n_clicks')
              ])
def displayClick(L_value,submit):
    changed_id=[p['prop_id'] for p in callback_context.triggered][0]
    if 'sub' in changed_id:
        url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
        df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')
        #data curation
        box=go.Figure()
        box=px.box(df,x='target_id',y='L*')
        box=box.add_trace(go.Scatter(x=['Blanc de Noir'],y=[L_value],showlegend=False))
        box=box.add_trace(go.Scatter(x=['Not Blanc de Noir'],y=[L_value],showlegend=False))
        box.update_layout(
            title="Boxplot L*-value",
            xaxis_title="Target",
            yaxis_title="L*",
            legend_title="Classification",
            font=dict(
                size=18,
            )
        )
    
        return box
    else:
        url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
        df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')
        box=px.box(df,x='target_id',y='L*')
        box.update_layout(
            title="Boxplot L*-value",
            xaxis_title="Target",
            yaxis_title="L*",
            legend_title="Classification",
            font=dict(
                size=18,
            )
        )
        return box
    
@app.callback(Output('boxa', 'figure'),
              [
                  State('a*','value'),
                  Input('sub','n_clicks')
              ])
def displayClick(a_value,submit):
    changed_id=[p['prop_id'] for p in callback_context.triggered][0]
    if 'sub' in changed_id:
        url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
        df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')
        #data curation
        box=go.Figure()
        box=px.box(df,x='target_id',y='a*')
        box=box.add_trace(go.Scatter(x=['Blanc de Noir'],y=[a_value],showlegend=False))
        box=box.add_trace(go.Scatter(x=['Not Blanc de Noir'],y=[a_value],showlegend=False))
        box.update_layout(
            title="Boxplot a*-value",
            xaxis_title="Target",
            yaxis_title="a*",
            legend_title="Classification",
            font=dict(
                size=18,
            )
        )
        return box
    else:
        url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
        df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')
        box=px.box(df,x='target_id',y='a*')
        box.update_layout(
            title="Boxplot a*-value",
            xaxis_title="Target",
            yaxis_title="a*",
            legend_title="Classification",
            font=dict(
                size=18,
            )
        )
        return box
    
@app.callback(Output('boxb', 'figure'),
              [
                  State('b*','value'),
                  Input('sub','n_clicks')
              ])
def displayClick(b_value,submit):
    changed_id=[p['prop_id'] for p in callback_context.triggered][0]
    if 'sub' in changed_id:
        url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
        df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')
        #data curation
        box=go.Figure()
        box=px.box(df,x='target_id',y='b*')
        box=box.add_trace(go.Scatter(x=['Blanc de Noir'],y=[b_value],showlegend=False))
        box=box.add_trace(go.Scatter(x=['Not Blanc de Noir'],y=[b_value],showlegend=False))
        box.update_layout(
            title="Boxplot b*-value",
            xaxis_title="Target",
            yaxis_title="b*",
            legend_title="Classification",
            font=dict(
                size=18,
            )
        )
    
        return box
    else:
        url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
        df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')
        box=px.box(df,x='target_id',y='b*')
        box.update_layout(
            title="Boxplot b*-value",
            xaxis_title="Target",
            yaxis_title="b*",
            legend_title="Classification",
            font=dict(
                size=18,
            )
        )
        return box
        
@app.callback(Output('3d','figure'),
              [State('L*','value'),
               State('a*','value'),
               State('b*','value'),
               Input('sub','n_clicks')
              ])
def displayClick(L_value,a_value,b_value,submit):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'sub' in changed_id:
        try:        
            #data import
            url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
            df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')
            y_id=df['target_id']
            y=df['target']
            X_data = df.drop(['target','target_id'], axis=1) 
            #scaling the data
            scaler = RobustScaler() 
            X_scale = scaler.fit_transform(X_data)
            X_scale_plot=pd.DataFrame(X_scale, columns=['L*','a*','b*'])
            # prepating the plot   
            X= X_scale_plot[['L*','a*','b*']]
            # add new data from Input boxes
            q1L,q3L=np.percentile(df['L*'],[25,75]) #fetch 25% and 75% quantils for l*
            IQRL=q3L-q1L # calculate Interquantilerange for L*
            scaled_new_L= (L_value-df['L*'].median(axis=0))/IQRL # Scaling the new data without changing the model
            q1a,q3a=np.percentile(df['a*'],[25,75]) ##fetch 25% and 75% quantils for a*
            IQRa=q3a-q1a  # calculate Interquantilerange for a*
            scaled_new_a=(a_value-df['a*'].median(axis=0))/IQRa # Scaling the new data without changing the model
            q1b,q3b = np.percentile(df['b*'],[25,75]) #fetch 25% and 75% quantils for b*
            IQRb = q3b-q1b # calculate Interquantilerange for b*
            scaled_new_b=(b_value-df['b*'].median(axis=0))/IQRb # Scaling the new data without changing the model
            new_data=[scaled_new_L,scaled_new_a,scaled_new_b] # Pass it to list
            # Computing SVM model with three variables 
            model =svm.SVC(kernel='rbf', gamma=0.8, C=21,) 
            clf = model.fit(X, y)
            url2=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/plotting_reference.csv"
            target_3f=pd.read_csv(url2, on_bad_lines='skip', delimiter=';')
            X0= pd.concat([X_scale_plot,target_3f[['target','target_id']]],axis=1)
            X_01= X0.query("target == 0")
            X_02= X0.query("target == 2")
            X_01.columns = X_01.columns.str.replace(' ', '')
            y_01=X_01['target']
            y_02=X_02['target']

            # Define mesh grid for plotting decision boundary
            xx, yy, zz = np.meshgrid(np.linspace(X['L*'].min() - 1, X['L*'].max() + 1, 50),
                                     np.linspace(X['a*'].min() - 1, X['a*'].max() + 1, 50),
                                     np.linspace(X['b*'].min() - 1, X['b*'].max() + 1, 50))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

            # Create plotly figure
            fig = go.Figure()
            fig.add_trace(go.Volume(x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), value=Z, isomin=0.5, isomax=0.5,
                                     opacity=0.01, surface_count=50, showscale=False,name='Decision Boundary',colorscale='gray'))
            fig.add_trace(go.Scatter3d(x=X_01['L*'], y=X_01['a*'], z=X_02['b*'],name='Reference white wine', mode='markers',
                                       marker=dict(color='blue', size=5, showscale=False),))
            fig.add_trace(go.Scatter3d(x=X_02['L*'], y=X_02['a*'], z=X_02['b*'],name='Reference Rosé', mode='markers',
                                       marker=dict(color='green', size=5, showscale=False),))
            # fig.add_trace(go.Scatter3d( x=clf.support_vectors_[:,0],y=clf.support_vectors_[:,1],z=clf.support_vectors_[:,2],name='Support Vector',mode='markers',
            #                            marker=dict(color='black', opacity=0.5, size=8,showscale=False)))
            fig.update_layout(
                autosize=False,
                width=1000,
                height=1000,colorscale=None)
            # Set layout properties
            fig.update_layout(scene=dict(xaxis_title='Standardized L*', yaxis_title='Standardized a*', zaxis_title='Standardized b*',
                                         xaxis = dict(range=[-5,1.8],),
                                  yaxis = dict(range=[-2.721095571095571,5]),
                                  zaxis = dict(range=[-1,5]),))
            fig.update_traces(showlegend=True)

            fig.add_trace(go.Scatter3d(x=[new_data[0]], y=[new_data[1]], z=[new_data[2]], name='New Wine',mode='markers',
                           marker=dict(color='red', size=5, showscale=False),))
            return fig
        except:
            raise PreventUpdate
    else:
        #data Import
        url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
        df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')
        #data curation
        y_id=df['target_id']
        y=df['target']
        X_data = df.drop(['target','target_id'], axis=1) 
        #scaling the data 
        scaler = RobustScaler() 
        X_scale = scaler.fit_transform(X_data)
        X_scale_plot=pd.DataFrame(X_scale, columns=['L*','a*','b*'])
        X= X_scale_plot[['L*','a*','b*']]
        ############## Computing three dimension ################################
        
        model =svm.SVC(kernel='rbf', gamma=0.8, C=21,)
        clf = model.fit(X, y)
        url2=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/plotting_reference.csv"
        target_3f=pd.read_csv(url2, on_bad_lines='skip', delimiter=';')
        X0= pd.concat([X_scale_plot,target_3f[['target','target_id']]],axis=1)
        X_01= X0.query("target == 0")
        X_02= X0.query("target == 2")
        X_01.columns = X_01.columns.str.replace(' ', '')
        y_01=X_01['target']
        y_02=X_02['target']
        # Define mesh grid for plotting decision boundary in three dimensions
        xx, yy, zz = np.meshgrid(np.linspace(X['L*'].min() - 1, X['L*'].max() + 1, 50),
                                 np.linspace(X['a*'].min() - 1, X['a*'].max() + 1, 50),
                                 np.linspace(X['b*'].min() - 1, X['b*'].max() + 1, 50))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

        # Create plotly figure
        fig = go.Figure()
        fig.add_trace(go.Volume(x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), value=Z, isomin=0.5, isomax=0.5,
                                 opacity=0.01, surface_count=50, showscale=False,name='Decision Boundary',colorscale='gray'))
        fig.add_trace(go.Scatter3d(x=X_01['L*'], y=X_01['a*'], z=X_02['b*'],name='Reference white wine', mode='markers',
                                   marker=dict(color='blue', size=5, showscale=False),))
        fig.add_trace(go.Scatter3d(x=X_02['L*'], y=X_02['a*'], z=X_02['b*'],name='Reference Rosé', mode='markers',
                                   marker=dict(color='green', size=5, showscale=False),))
        # fig.add_trace(go.Scatter3d( x=clf.support_vectors_[:,0],y=clf.support_vectors_[:,1],z=clf.support_vectors_[:,2],name='Support Vector',mode='markers',
        #                            marker=dict(color='black', opacity=0.5, size=8,showscale=False)))
        fig.update_layout(
            autosize=False,
            width=1000,
            height=1000,colorscale=None)
        # Set layout properties
        fig.update_layout(scene=dict(xaxis_title='Standardized L*', yaxis_title='Standardized a*', zaxis_title='Standardized b*', 
                                     xaxis = dict(range=[-5,1.8],),
                              yaxis = dict(range=[-2.721095571095571,5]),
                              zaxis = dict(range=[-1,5]),), title_text='L*a*b* projection')
        fig.update_traces(showlegend=True)

    
    return fig
@app.callback(Output('Laplot','figure'),
              [State('L*','value'),
               State('a*','value'),
               State('b*','value'),
               Input('sub','n_clicks')
              ])
def displayClick(L_value,a_value,b_value,submit):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'sub' in changed_id:
        try:
            url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
            df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')# Pandas DataFrame aus Exceldatei
            y_id=df['target_id']
            X_data = df.drop(['target','target_id'], axis=1) # X = alle Variablen außer das target
            y = df['target'] 
            scaler = RobustScaler() 
            X_scale = scaler.fit_transform(X_data)
            X_scale_plot=pd.DataFrame(X_scale, columns=['L*','a*','b*'])
            X_train, X_test, y_train, y_test = train_test_split(X_scale,y,stratify=y,test_size=0.3, random_state=10)
            X_la= X_scale_plot[['L*','a*']]
            url2=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/plotting_reference.csv"
            target_3f=pd.read_csv(url2, on_bad_lines='skip', delimiter=';')
            X0= pd.concat([X_scale_plot,target_3f[['target','target_id']]],axis=1)
            X_01= X0.query("target == 0")
            X_02= X0.query("target == 2")
            X_01.columns = X_01.columns.str.replace(' ', '')
            y_01=X_01['target']
            y_02=X_02['target']

            # Fit SVM model
            model =svm.SVC(kernel='rbf', gamma=0.8, C=21,)
            clf = model.fit(X_la, y)
            x_min, x_max = X_la['L*'].min() - 1, X_la['L*'].max() + 1
            y_min, y_max = X_la['a*'].min() - 1, X_la['a*'].max() + 1
            xx2, yy2 = np.meshgrid(np.linspace(x_min, x_max, 1000),
                                 np.linspace(y_min, y_max, 1000))
            Z2 = clf.predict(np.c_[xx2.ravel(), yy2.ravel()])
            Z2 = Z2.reshape(xx2.shape)
            ########## Plotting decision boundaries###########################
            q1L,q3L=np.percentile(df['L*'],[25,75]) #fetch 25% and 75% quantils for l*
            IQRL=q3L-q1L # calculate Interquantilerange for L*
            scaled_new_L= (L_value-df['L*'].median(axis=0))/IQRL # Scaling the new data without changing the model
            q1a,q3a=np.percentile(df['a*'],[25,75]) ##fetchStandardized  25% and 75% quantils for *
            IQRa=q3a-q1a  # calculate InterquantilStandardized erange for a*
            scaled_new_a=(a_value-df['a*'].median(axis=0))/IQRa # Scaling the new data without changing the model
            q1b,q3b = np.percentile(df['b*'],[25,75]) #fetch 25% and 75% quantils for b*
            IQRb = q3b-q1b # calculate Interquantilerange for b*
            scaled_new_b=(b_value-df['b*'].median(axis=0))/IQRb # Scaling the new data without changing the model
            new_data=[scaled_new_L,scaled_new_a,scaled_new_b] # Pass it to list
            # Fit SVM model

            # # create grid to evaluate model
            ######### Creating four different graphs for #####################
            figla = go.Figure() # Creating L*a* projection 2D
            figla.add_trace(go.Contour(x=xx2[0], y=yy2[:, 0],
                               z=Z2,contours_coloring='lines',
                               colorscale='ylorrd_r',
                               showscale=False,
                               opacity=.8,
                               line_smoothing=True))
            figla.add_trace(go.Scatter(x=X_01['L*'], y=X_01['a*'], name='Reference white wine', mode='markers',
                                       marker=dict(color='blue', size=5, showscale=False),))
            figla.add_trace(go.Scatter(x=X_02['L*'], y=X_02['a*'], name='Reference Rosé', mode='markers',
                                       marker=dict(color='green', size=5, showscale=False),))
            figla.add_trace(go.Scatter(x=[new_data[0]], y=[new_data[1]], name='New Wine',mode='markers',
                           marker=dict(color='red', size=5, showscale=False),))
            # 
            figla.add_annotation(x=1, y=.8,
                        text="Blanc de Noir",
                        showarrow=False,
                        yshift=10,font=dict(size=20))
            figla.add_annotation(x=-3, y=4,
                        text="Not Blanc de Noir",
                        showarrow=False,
                        yshift=10,font=dict(size=20,color='black'))
            figla.update_layout(
                autosize=False,
                colorscale=None,
                plot_bgcolor='white',
                paper_bgcolor='white',title='L*a* projection',
                xaxis_title='Standardized L*',
                yaxis_title='Standardized a*',
            )
            figla.update_xaxes(range=[-5,1.8],showgrid=False)
            figla.update_yaxes(range=[-2.721095571095571,5],showgrid=False)
            return figla
        except:
            raise PreventUpdate
    else:
        url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
        df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')# Pandas DataFrame aus Exceldatei
        y_id=df['target_id']
        X_data = df.drop(['target','target_id'], axis=1) # X = alle Variablen außer das target
        y = df['target'] 
        scaler = RobustScaler() 
        X_scale = scaler.fit_transform(X_data)
        X_scale_plot=pd.DataFrame(X_scale, columns=['L*','a*','b*'])
        X_train, X_test, y_train, y_test = train_test_split(X_scale,y,stratify=y,test_size=0.3, random_state=10)
        url2=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/plotting_reference.csv"
        target_3f=pd.read_csv(url2, on_bad_lines='skip', delimiter=';')
        X0= pd.concat([X_scale_plot,target_3f[['target','target_id']]],axis=1)
        X_la= X_scale_plot[['L*','a*']]
        X_01= X0.query("target == 0")
        X_02= X0.query("target == 2")
        X_01.columns = X_01.columns.str.replace(' ', '')
        y_01=X_01['target']
        y_02=X_02['target']
        # Fit SVM model
        model =svm.SVC(kernel='rbf', gamma=0.8, C=21,)
        clf = model.fit(X_la, y)
        x_min, x_max = X_la['L*'].min() - 1, X_la['L*'].max() + 1
        y_min, y_max = X_la['a*'].min() - 1, X_la['a*'].max() + 1
        xx2, yy2 = np.meshgrid(np.linspace(x_min, x_max, 1000),
                             np.linspace(y_min, y_max, 1000))
        Z2 = clf.predict(np.c_[xx2.ravel(), yy2.ravel()])
        Z2 = Z2.reshape(xx2.shape)
        # Fit SVM model
        
        # # create grid to evaluate model
        ######### Creating four different graphs for #####################
        figla = go.Figure() # Creating L*a* projection 2D
        figla.add_trace(go.Contour(x=xx2[0], y=yy2[:, 0],
                           z=Z2, contours_coloring='lines',
                           colorscale='ylorrd_r',
                           showscale=False,
                           opacity=.8,
                           line_smoothing=True))
        figla.add_trace(go.Scatter(x=X_01['L*'], y=X_01['a*'], name='Reference white wine', mode='markers',
                                   marker=dict(color='blue', size=5, showscale=False),))
        figla.add_trace(go.Scatter(x=X_02['L*'], y=X_02['a*'], name='Reference Rosé', mode='markers',
                                   marker=dict(color='green', size=5, showscale=False),))

        # 
        figla.add_annotation(x=1, y=0.8,
                    text="Blanc de Noir",
                    showarrow=False,
                    yshift=10,font=dict(size=16))
        figla.add_annotation(x=-3, y=4,
                    text="Not Blanc de Noir",
                    showarrow=False,
                    yshift=10,font=dict(size=16,color='black'))
        figla.update_layout(
            autosize=False,
            colorscale=None,
            plot_bgcolor='white',
            paper_bgcolor='white',title='L*a* projection',
            xaxis_title='Standardized L*',
            yaxis_title='Standardized a*',
        )
        figla.update_xaxes(range=[-5,1.8],showgrid=False)
        figla.update_yaxes(range=[-2.721095571095571,5],showgrid=False)
        return figla
@app.callback(Output('Lbplot','figure'),
              [State('L*','value'),
               State('a*','value'),
               State('b*','value'),
               Input('sub','n_clicks')
              ])
def displayClick(L_value,a_value,b_value,submit):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'sub' in changed_id:
        try:
            url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
            df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')# Pandas DataFrame aus Exceldatei
            y_id=df['target_id']
            X_data = df.drop(['target','target_id'], axis=1) # X = alle Variablen außer das target
            y = df['target'] 
            scaler = RobustScaler() 
            X_scale = scaler.fit_transform(X_data)
            X_scale_plot=pd.DataFrame(X_scale, columns=['L*','a*','b*'])
            X_train, X_test, y_train, y_test = train_test_split(X_scale,y,stratify=y,test_size=0.3, random_state=10)
            X_lb= X_scale_plot[['L*','b*']]
            url2=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/plotting_reference.csv"
            target_3f=pd.read_csv(url2, on_bad_lines='skip', delimiter=';')
            X0= pd.concat([X_scale_plot,target_3f[['target','target_id']]],axis=1)
            X_la= X_scale_plot[['L*','a*']]
            X_01= X0.query("target == 0")
            X_02= X0.query("target == 2")
            X_01.columns = X_01.columns.str.replace(' ', '')
            y_01=X_01['target']
            y_02=X_02['target']

            # Fit SVM model
            model =svm.SVC(kernel='rbf', gamma=0.8, C=21,)
            clf = model.fit(X_lb, y)
            x_min, x_max = X_lb['L*'].min() - 1, X_lb['L*'].max() + 1
            y_min, y_max = X_lb['b*'].min() - 1, X_lb['b*'].max() + 1
            xx2, yy2 = np.meshgrid(np.linspace(x_min, x_max, 1000),
                                 np.linspace(y_min, y_max, 1000))
            Z2 = clf.predict(np.c_[xx2.ravel(), yy2.ravel()])
            Z2 = Z2.reshape(xx2.shape)
            ########## Plotting decision boundaries###########################
            q1L,q3L=np.percentile(df['L*'],[25,75]) #fetch 25% and 75% quantils for l*
            IQRL=q3L-q1L # calculate Interquantilerange for L*
            scaled_new_L= (L_value-df['L*'].median(axis=0))/IQRL # Scaling the new data without changing the model
            q1a,q3a=np.percentile(df['a*'],[25,75]) #Standardized #fetch 25% and 75% quantils for *
            IQRa=q3a-q1a  # calculate InterqStandardized uantilerange for a*
            scaled_new_a=(a_value-df['a*'].median(axis=0))/IQRa # Scaling the new data without changing the model
            q1b,q3b = np.percentile(df['b*'],[25,75]) #fetch 25% and 75% quantils for b*
            IQRb = q3b-q1b # calculate Interquantilerange for b*
            scaled_new_b=(b_value-df['b*'].median(axis=0))/IQRb # Scaling the new data without changing the model
            new_data=[scaled_new_L,scaled_new_a,scaled_new_b] # Pass it to list
            # Fit SVM model

            # # create grid to evaluate model
            ######### Creating four different graphs for #####################
            figlb = go.Figure() # Creating L*a* projection 2D
            figlb.add_trace(go.Contour(x=xx2[0], y=yy2[:, 0],
                               z=Z2, contours_coloring='lines',
                               colorscale='ylorrd_r',
                               showscale=False,
                               opacity=.8,
                               line_smoothing=True))
            figlb.add_trace(go.Scatter(x=[new_data[0]], y=[new_data[1]], name='New Wine',mode='markers',
                           marker=dict(color='red', size=5, showscale=False),))
            figlb.add_trace(go.Scatter(x=X_01['L*'], y=X_01['b*'], name='Reference white wine', mode='markers',
                                       marker=dict(color='blue', size=5, showscale=False),))
            figlb.add_trace(go.Scatter(x=X_02['L*'], y=X_02['b*'], name='Reference Rosé', mode='markers',
                                       marker=dict(color='green', size=5, showscale=False),))

            # 
            figlb.add_annotation(x=-.75, y=-.9,  text="Blanc de Noir",
                        showarrow=False,
                        yshift=10,font=dict(size=16))
            figlb.add_annotation(x=-3, y=3,
                        text="Not Blanc de Noir",
                        showarrow=False,
                        yshift=10,font=dict(size=16,color='black'))
            figlb.update_layout(
                autosize=False,
                colorscale=None,
                plot_bgcolor='white',
                paper_bgcolor='white',title='L*b* projection',
                xaxis_title='Standardized L*',
                yaxis_title='Standardized b*',
            )
            figlb.update_xaxes(range=[-5,1.8],showgrid=False)
            figlb.update_yaxes(range=[-1,4],showgrid=False)

            return figlb
        except:
            raise PreventUpdate
        
    else:
        url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
        df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')# Pandas DataFrame aus Exceldatei
        y_id=df['target_id']
        X_data = df.drop(['target','target_id'], axis=1) # X = alle Variablen außer das target
        y = df['target'] 
        scaler = RobustScaler() 
        X_scale = scaler.fit_transform(X_data)
        X_scale_plot=pd.DataFrame(X_scale, columns=['L*','a*','b*'])
        X_train, X_test, y_train, y_test = train_test_split(X_scale,y,stratify=y,test_size=0.3, random_state=10)
        X_lb= X_scale_plot[['L*','b*']]
        X_la= X_scale_plot[['L*','a*']]
        url2=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/plotting_reference.csv"
        target_3f=pd.read_csv(url2, on_bad_lines='skip', delimiter=';')
        X0= pd.concat([X_scale_plot,target_3f[['target','target_id']]],axis=1)
        X_01= X0.query("target == 0")
        X_02= X0.query("target == 2")
        X_01.columns = X_01.columns.str.replace(' ', '')
        y_01=X_01['target']
        y_02=X_02['target']
        # Fit SVM mStandardized odel
        model =svm.SVC(kernel='rbf', gamma=0.8, C=21,)
        clf = model.fit(X_lb, y)
        x_min, x_max = X_lb['L*'].min() - 1, X_lb['L*'].max() + 1
        y_min, y_max = X_lb['b*'].min() - 1, X_lb['b*'].max() + 1
        xx2, yy2 = np.meshgrid(np.linspace(x_min, x_max, 1000),
                             np.linspace(y_min, y_max, 1000))
        Z2 = clf.predict(np.c_[xx2.ravel(), yy2.ravel()])
        Z2 = Z2.reshape(xx2.shape)
        # Fit SVM model
        
        # # create grid to evaluate model
        ######### Creating four different graphs for #####################
        figlb = go.Figure() # Creating L*a* projection 2D
        figlb.add_trace(go.Contour(x=xx2[0], y=yy2[:, 0], 
                           z=Z2,contours_coloring='lines',
                           colorscale='ylorrd_r',
                           showscale=False,
                           opacity=.8,
                           line_smoothing=True))
        figlb.add_trace(go.Scatter(x=X_01['L*'], y=X_01['b*'], name='Reference white wine', mode='markers',
                                   marker=dict(color='blue', size=5, showscale=False),))
        figlb.add_trace(go.Scatter(x=X_02['L*'], y=X_02['b*'], name='Reference Rosé', mode='markers',
                                   marker=dict(color='green', size=5, showscale=False),))
        # 
        figlb.add_annotation(x=-.75, y=-.9,
                    text="Blanc de Noir",
                    showarrow=False,
                    yshift=10,font=dict(size=12))
        figlb.add_annotation(x=-3, y=3,
                    text="Not Blanc de Noir",
                    showarrow=False,
                    yshift=10,font=dict(size=16,color='black'))
        figlb.update_layout(
            autosize=False,
            colorscale=None,
            plot_bgcolor='white',
            paper_bgcolor='white',title='L*b* projection',
            xaxis_title='Standardized L*',
            yaxis_title='Standardized b*',
        )
        figlb.update_xaxes(range=[-5,1.8],showgrid=False)
        figlb.update_yaxes(range=[-1,4],showgrid=False)
        
        return figlb
@app.callback(Output('abplot','figure'),
              [State('L*','value'),
               State('a*','value'),
               State('b*','value'),
               Input('sub','n_clicks')
              ])
def displayClick(L_value,a_value,b_value,submit):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'sub' in changed_id:
        try:
            url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
            df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')# Pandas DataFrame aus Exceldatei
            y_id=df['target_id']
            X_data = df.drop(['target','target_id'], axis=1) # X = alle Variablen außer das target
            y = df['target'] 
            scaler = RobustScaler() 
            X_scale = scaler.fit_transform(X_data)
            X_scale_plot=pd.DataFrame(X_scale, columns=['L*','a*','b*'])
            X_train, X_test, y_train, y_test = train_test_split(X_scale,y,stratify=y,test_size=0.3, random_state=10)
            X_ab= X_scale_plot[['a*','b*']]
            url2=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/plotting_reference.csv"
            target_3f=pd.read_csv(url2, on_bad_lines='skip', delimiter=';')
            X0= pd.concat([X_scale_plot,target_3f[['target','target_id']]],axis=1)
            X_01= X0.query("target == 0")
            X_02= X0.query("target == 2")
            X_01.columns = X_01.columns.str.replace(' ', '')
            y_01=X_01['target']
            y_02=X_02['target']

            # Fit SVM model
            model =svm.SVC(kernel='rbf', gamma=0.8, C=21,)
            clf = model.fit(X_ab, y)
            x_min, x_max = X_ab['a*'].min() - 1, X_ab['a*'].max() + 1
            y_min, y_max = X_ab['b*'].min() - 1, X_ab['b*'].max() + 1
            xx3, yy3 = np.meshgrid(np.linspace(x_min, x_max, 1000),
                                 np.linspace(y_min, y_max,1000))
            Z3 = clf.predict(np.c_[xx3.ravel(), yy3.ravel()])
            Z3 = Z3.reshape(xx3.shape)
            ########## Plotting decision boundaries###########################
            q1L,q3L=np.percentile(df['L*'],[25,75]) #fetch 25% and 75% quantils for l*
            IQRL=q3L-q1L # calculate Interquantilerange for L*
            scaled_new_L= (L_value-df['L*'].median(axis=0))/IQRL # Scaling the new data without changing the model
            q1a,q3a=np.percentile(df['a*'],[25,75]) ##fetch 25% and 75% quantils for *
            IQRa=q3a-q1a  # calculate Interquantilerange for a*
            scaled_new_a=(a_value-df['a*'].median(axis=0))/IQRa # Scaling the new data without changing the model
            q1b,q3b = np.percentile(df['b*'],[25,75]) #fetch 25% and 75% quantils for b*
            IQRb = q3b-q1b # calculate Interquantilerange for b*
            scaled_new_b=(b_value-df['b*'].median(axis=0))/IQRb # Scaling the new data without changing the model
            new_data=[scaled_new_L,scaled_new_a,scaled_new_b] # Pass it to list
            # Fit SVM model

            # # create grid to evaluate model
            ######### Creating four different graphs for #####################
            figab = go.Figure() # Creating L*a* projection 2D
            figab.add_trace(go.Contour(x=xx3[0], y=yy3[:, 0],
                               z=Z3,
                               colorscale='ylorrd_r', contours_coloring='lines',
                               showscale=False,
                               opacity=.8,
                               line_smoothing=True))
            figab.add_trace(go.Scatter(x=[new_data[0]], y=[new_data[1]], name='New Wine',mode='markers',
                           marker=dict(color='red', size=5, showscale=False),))
            figab.add_trace(go.Scatter(x=X_01['a*'], y=X_01['b*'], name='Reference white wine', mode='markers',
                                       marker=dict(color='blue', size=5, showscale=False),))
            figab.add_trace(go.Scatter(x=X_02['a*'], y=X_02['b*'], name='Reference Rosé', mode='markers',
                                       marker=dict(color='green', size=5, showscale=False),))
            # 
            figab.add_annotation(x=0.5, y=-0.8,
                text="Blanc de Noir",
                showarrow=False,
                yshift=10,font=dict(size=20))
            figab.add_annotation(x=-1.8, y=3.5,
                text="Not Blanc de Noir",
                showarrow=False,
                yshift=10,font=dict(size=20,color='black'))

            figab.update_layout(
                autosize=False,
                colorscale=None,
                plot_bgcolor='white',
                paper_bgcolor='white',title='a*b* projection',
                xaxis_title='Standardized a*',
                yaxis_title='Standardized b',
            )
            figab.update_xaxes(range=[-2.5,1.8],showgrid=False)
            figab.update_yaxes(range=[-2.5,4],showgrid=False)

            return figab
        except:
            raise PreventUpdate
        
    else:
        url=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/Blanc_de_Noir_check_training_data.csv"
        df = pd.read_csv(url,on_bad_lines='skip', delimiter=',')# Pandas DataFrame aus Exceldatei
        y_id=df['target_id']
        X_data = df.drop(['target','target_id'], axis=1) # X = alle Variablen außer das target
        y = df['target'] 
        scaler = RobustScaler() 
        X_scale = scaler.fit_transform(X_data)
        X_scale_plot=pd.DataFrame(X_scale, columns=['L*','a*','b*'])
        X_train, X_test, y_train, y_test = train_test_split(X_scale,y,stratify=y,test_size=0.3, random_state=10)
        X_ab= X_scale_plot[['a*','b*']]
        url2=r"https://raw.githubusercontent.com/Der-Hensel/Blanc_de_Noir_check/main/plotting_reference.csv"
        target_3f=pd.read_csv(url2, on_bad_lines='skip', delimiter=';')
        X0= pd.concat([X_scale_plot,target_3f[['target','target_id']]],axis=1)
        X_01= X0.query("target == 0")
        X_02= X0.query("target == 2")
        X_01.columns = X_01.columns.str.replace(' ', '')
        y_01=X_01['target']
        y_02=X_02['target']
        
        
        # Fit SVM model
        model =svm.SVC(kernel='rbf', gamma=0.8, C=21,)
        clf = model.fit(X_ab, y)
        x_min, x_max = X_ab['a*'].min() - 1, X_ab['a*'].max() + 1
        y_min, y_max = X_ab['b*'].min() - 1, X_ab['b*'].max() + 1
        xx3, yy3 = np.meshgrid(np.linspace(x_min, x_max, 1000),
                             np.linspace(y_min, y_max, 1000))
        Z3 = clf.predict(np.c_[xx3.ravel(), yy3.ravel()])
        Z3 = Z3.reshape(xx3.shape)
        # Fit SVM model
        
        # # create grid to evaluate mStandardized odel
        ######### Creating four differStandardized ent graphs for #####################
        figab = go.Figure() # Creating L*a* projection 2D
        figab.add_trace(go.Contour(x=xx3[0], y=yy3[:, 0],
                           z=Z3,contours_coloring='lines',
                           colorscale='ylorrd_r',
                           showscale=False,
                           opacity=.8,
                           line_smoothing=True))
        figab.add_trace(go.Scatter(x=X_01['a*'], y=X_01['b*'], name='Reference white wine', mode='markers',
                                   marker=dict(color='blue', size=5, showscale=False),))
        figab.add_trace(go.Scatter(x=X_02['a*'], y=X_02['b*'], name='Reference Rosé', mode='markers',
                                   marker=dict(color='green', size=5, showscale=False),))
        # 
        figab.add_annotation(x=0.5, y=-0.8,
            text="Blanc de Noir",
            showarrow=False,
            yshift=10,font=dict(size=20))
        figab.add_annotation(x=-1.8, y=3.5,
            text="Not Blanc de Noir",
            showarrow=False,
            yshift=10,font=dict(size=20,color='black'))

        figab.update_layout(
            autosize=False,
            colorscale=None,
            plot_bgcolor='white',
            paper_bgcolor='white',title='a*b* projection',
            xaxis_title='Standardized a*-value',
            yaxis_title='Standardized b*-value'
        )
        figab.update_xaxes(range=[-2.5,1.8],showgrid=False)
        figab.update_yaxes(range=[-2.5,4],showgrid=False)
        
        return figab

if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False, port=8000)

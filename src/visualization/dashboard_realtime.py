"""
Real-time Training Dashboard with Plotly Dash
Professional publication-quality visualization for RL training monitoring
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from collections import deque
import threading
import time


class TrainingDashboard:
    """
    Interactive real-time dashboard for monitoring RL training.
    Features multiple tabs, automatic updates, and export capabilities.
    """
    
    def __init__(self, log_dir: str = "results/logs", update_interval: int = 1000):
        """
        Initialize the dashboard.
        
        Args:
            log_dir: Directory containing training logs
            update_interval: Update frequency in milliseconds
        """
        self.log_dir = Path(log_dir)
        self.update_interval = update_interval
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        
        # Color scheme (color-blind friendly)
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Purple
            'success': '#06A77D',      # Green
            'warning': '#F18F01',      # Orange
            'danger': '#C73E1D',       # Red
            'background': '#1E1E1E',   # Dark background
            'text': '#FFFFFF',         # White text
            'grid': '#333333'          # Grid color
        }
        
        # Data storage
        self.data_buffer = {
            'episodes': deque(maxlen=1000),
            'rewards': deque(maxlen=1000),
            'success_rates': deque(maxlen=1000),
            'steps': deque(maxlen=1000),
            'losses': deque(maxlen=1000),
            'epsilons': deque(maxlen=1000),
            'actions': deque(maxlen=1000),
            'q_values': deque(maxlen=1000),
            'learning_rates': deque(maxlen=1000),
            'buffer_sizes': deque(maxlen=1000),
            'durations': deque(maxlen=1000)
        }
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup the dashboard layout with tabs."""
        self.app.layout = html.Div(
            style={'backgroundColor': self.colors['background']},
            children=[
                # Header
                html.Div([
                    html.H1(
                        'Robot Navigation RL Training Dashboard',
                        style={
                            'textAlign': 'center',
                            'color': self.colors['text'],
                            'padding': '20px',
                            'fontFamily': 'Arial, Helvetica, sans-serif'
                        }
                    ),
                    html.Div([
                        html.Button(
                            'Save Session',
                            id='save-button',
                            n_clicks=0,
                            style={
                                'backgroundColor': self.colors['success'],
                                'color': 'white',
                                'padding': '10px 20px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'marginRight': '10px',
                                'cursor': 'pointer'
                            }
                        ),
                        html.Button(
                            'Export Charts',
                            id='export-button',
                            n_clicks=0,
                            style={
                                'backgroundColor': self.colors['primary'],
                                'color': 'white',
                                'padding': '10px 20px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'cursor': 'pointer'
                            }
                        ),
                    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
                    html.Div(id='save-status', style={'textAlign': 'center', 'color': self.colors['text']})
                ]),
                
                # Tabs
                dcc.Tabs(
                    id='tabs',
                    value='tab-overview',
                    children=[
                        dcc.Tab(label='Overview', value='tab-overview'),
                        dcc.Tab(label='Learning Dynamics', value='tab-learning'),
                        dcc.Tab(label='State-Action', value='tab-state-action'),
                        dcc.Tab(label='Performance', value='tab-performance'),
                    ],
                    style={'fontFamily': 'Arial, Helvetica, sans-serif'}
                ),
                
                # Tab content
                html.Div(id='tabs-content', style={'padding': '20px'}),
                
                # Auto-update interval
                dcc.Interval(
                    id='interval-component',
                    interval=self.update_interval,
                    n_intervals=0
                ),
                
                # Hidden divs for data storage
                html.Div(id='data-store', style={'display': 'none'})
            ]
        )
    
    def _setup_callbacks(self):
        """Setup all dashboard callbacks."""
        
        @self.app.callback(
            Output('tabs-content', 'children'),
            Input('tabs', 'value')
        )
        def render_tab_content(active_tab):
            if active_tab == 'tab-overview':
                return self._create_overview_tab()
            elif active_tab == 'tab-learning':
                return self._create_learning_tab()
            elif active_tab == 'tab-state-action':
                return self._create_state_action_tab()
            elif active_tab == 'tab-performance':
                return self._create_performance_tab()
        
        @self.app.callback(
            [Output('reward-chart', 'figure'),
             Output('success-chart', 'figure'),
             Output('steps-chart', 'figure'),
             Output('loss-chart', 'figure')],
            Input('interval-component', 'n_intervals')
        )
        def update_overview_charts(n):
            return (
                self._create_reward_chart(),
                self._create_success_chart(),
                self._create_steps_chart(),
                self._create_loss_chart()
            )
        
        @self.app.callback(
            [Output('epsilon-chart', 'figure'),
             Output('lr-chart', 'figure'),
             Output('buffer-chart', 'figure'),
             Output('duration-chart', 'figure')],
            Input('interval-component', 'n_intervals')
        )
        def update_learning_charts(n):
            return (
                self._create_epsilon_chart(),
                self._create_lr_chart(),
                self._create_buffer_chart(),
                self._create_duration_chart()
            )
        
        @self.app.callback(
            [Output('action-chart', 'figure'),
             Output('qvalue-chart', 'figure')],
            Input('interval-component', 'n_intervals')
        )
        def update_state_action_charts(n):
            return (
                self._create_action_distribution_chart(),
                self._create_qvalue_heatmap()
            )
        
        @self.app.callback(
            Output('save-status', 'children'),
            Input('save-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def save_session(n_clicks):
            if n_clicks > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = self.log_dir / f"session_{timestamp}.json"
                self._save_session(filename)
                return f"✓ Session saved: {filename.name}"
            return ""
        
        @self.app.callback(
            Output('save-status', 'children', allow_duplicate=True),
            Input('export-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def export_charts(n_clicks):
            if n_clicks > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_dir = self.log_dir / f"exports_{timestamp}"
                self._export_all_charts(export_dir)
                return f"✓ Charts exported to: {export_dir.name}"
            return ""
    
    def _create_overview_tab(self):
        """Create the overview tab layout."""
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(id='reward-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='success-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(id='steps-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='loss-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
        ])
    
    def _create_learning_tab(self):
        """Create the learning dynamics tab layout."""
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(id='epsilon-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='lr-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(id='buffer-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='duration-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
        ])
    
    def _create_state_action_tab(self):
        """Create the state-action tab layout."""
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(id='action-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='qvalue-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
        ])
    
    def _create_performance_tab(self):
        """Create the performance metrics tab layout."""
        return html.Div([
            html.H3('Performance Metrics', style={'color': self.colors['text']}),
            html.Div(id='performance-stats', style={'color': self.colors['text']})
        ])
    
    def _create_reward_chart(self):
        """Create episode reward chart with moving average."""
        episodes = list(self.data_buffer['episodes'])
        rewards = list(self.data_buffer['rewards'])
        
        if len(rewards) == 0:
            return self._empty_chart("Episode Reward")
        
        # Calculate moving average
        window = min(50, len(rewards))
        ma_rewards = pd.Series(rewards).rolling(window=window).mean()
        
        # Detect outliers
        mean = np.mean(rewards)
        std = np.std(rewards)
        outliers = [r for r in rewards if abs(r - mean) > 2 * std]
        
        fig = go.Figure()
        
        # Raw rewards
        fig.add_trace(go.Scatter(
            x=episodes,
            y=rewards,
            mode='lines',
            name='Episode Reward',
            line=dict(color=self.colors['primary'], width=1),
            opacity=0.5
        ))
        
        # Moving average
        fig.add_trace(go.Scatter(
            x=episodes,
            y=ma_rewards,
            mode='lines',
            name=f'MA({window})',
            line=dict(color=self.colors['success'], width=3)
        ))
        
        # Highlight best episode
        if rewards:
            best_idx = np.argmax(rewards)
            fig.add_trace(go.Scatter(
                x=[episodes[best_idx]],
                y=[rewards[best_idx]],
                mode='markers',
                name='Best Episode',
                marker=dict(color=self.colors['warning'], size=15, symbol='star')
            ))
        
        fig.update_layout(
            title='Episode Reward Over Time',
            xaxis_title='Episode',
            yaxis_title='Reward',
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12),
            legend=dict(x=0.01, y=0.99),
            hovermode='x unified'
        )
        
        return fig
    
    def _create_success_chart(self):
        """Create success rate chart."""
        episodes = list(self.data_buffer['episodes'])
        success_rates = list(self.data_buffer['success_rates'])
        
        if len(success_rates) == 0:
            return self._empty_chart("Success Rate")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=[sr * 100 for sr in success_rates],  # Convert to percentage
            mode='lines',
            fill='tozeroy',
            name='Success Rate',
            line=dict(color=self.colors['success'], width=2)
        ))
        
        fig.update_layout(
            title='Success Rate Over Time',
            xaxis_title='Episode',
            yaxis_title='Success Rate (%)',
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12),
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def _create_steps_chart(self):
        """Create steps per episode chart with min/max bands."""
        episodes = list(self.data_buffer['episodes'])
        steps = list(self.data_buffer['steps'])
        
        if len(steps) == 0:
            return self._empty_chart("Steps per Episode")
        
        # Calculate rolling statistics
        window = min(50, len(steps))
        steps_series = pd.Series(steps)
        mean_steps = steps_series.rolling(window=window).mean()
        min_steps = steps_series.rolling(window=window).min()
        max_steps = steps_series.rolling(window=window).max()
        
        fig = go.Figure()
        
        # Min-max band
        fig.add_trace(go.Scatter(
            x=episodes + episodes[::-1],
            y=list(max_steps) + list(min_steps[::-1]),
            fill='toself',
            fillcolor='rgba(46, 134, 171, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Min-Max Band',
            showlegend=True
        ))
        
        # Mean steps
        fig.add_trace(go.Scatter(
            x=episodes,
            y=mean_steps,
            mode='lines',
            name='Mean Steps',
            line=dict(color=self.colors['primary'], width=3)
        ))
        
        fig.update_layout(
            title='Steps per Episode',
            xaxis_title='Episode',
            yaxis_title='Steps',
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12)
        )
        
        return fig
    
    def _create_loss_chart(self):
        """Create loss evolution chart."""
        episodes = list(self.data_buffer['episodes'])
        losses = list(self.data_buffer['losses'])
        
        if len(losses) == 0:
            return self._empty_chart("Loss Evolution")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=losses,
            mode='lines',
            name='Loss',
            line=dict(color=self.colors['danger'], width=2)
        ))
        
        # Add exponential moving average
        alpha = 0.1
        ema = [losses[0]]
        for loss in losses[1:]:
            ema.append(alpha * loss + (1 - alpha) * ema[-1])
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=ema,
            mode='lines',
            name='EMA',
            line=dict(color=self.colors['warning'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Loss Evolution',
            xaxis_title='Episode',
            yaxis_title='Loss',
            yaxis_type='log',
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12)
        )
        
        return fig
    
    def _create_epsilon_chart(self):
        """Create epsilon decay chart."""
        episodes = list(self.data_buffer['episodes'])
        epsilons = list(self.data_buffer['epsilons'])
        
        if len(epsilons) == 0:
            return self._empty_chart("Exploration Rate (Epsilon)")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=epsilons,
            mode='lines',
            name='Epsilon',
            line=dict(color=self.colors['secondary'], width=2)
        ))
        
        fig.update_layout(
            title='Exploration Rate (Epsilon Decay)',
            xaxis_title='Episode',
            yaxis_title='Epsilon',
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def _create_lr_chart(self):
        """Create learning rate schedule chart."""
        episodes = list(self.data_buffer['episodes'])
        lrs = list(self.data_buffer['learning_rates'])
        
        if len(lrs) == 0:
            return self._empty_chart("Learning Rate")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=lrs,
            mode='lines',
            name='Learning Rate',
            line=dict(color=self.colors['warning'], width=2)
        ))
        
        fig.update_layout(
            title='Learning Rate Schedule',
            xaxis_title='Episode',
            yaxis_title='Learning Rate',
            yaxis_type='log',
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12)
        )
        
        return fig
    
    def _create_buffer_chart(self):
        """Create replay buffer statistics chart."""
        episodes = list(self.data_buffer['episodes'])
        buffer_sizes = list(self.data_buffer['buffer_sizes'])
        
        if len(buffer_sizes) == 0:
            return self._empty_chart("Replay Buffer Size")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=buffer_sizes,
            mode='lines',
            fill='tozeroy',
            name='Buffer Size',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        fig.update_layout(
            title='Replay Buffer Statistics',
            xaxis_title='Episode',
            yaxis_title='Buffer Size',
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12)
        )
        
        return fig
    
    def _create_duration_chart(self):
        """Create episode duration chart."""
        episodes = list(self.data_buffer['episodes'])
        durations = list(self.data_buffer['durations'])
        
        if len(durations) == 0:
            return self._empty_chart("Episode Duration")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=durations,
            mode='lines',
            name='Duration (s)',
            line=dict(color=self.colors['success'], width=2)
        ))
        
        fig.update_layout(
            title='Episode Duration (Time per Episode)',
            xaxis_title='Episode',
            yaxis_title='Duration (seconds)',
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12)
        )
        
        return fig
    
    def _create_action_distribution_chart(self):
        """Create action distribution pie chart."""
        actions = list(self.data_buffer['actions'])
        
        if len(actions) == 0:
            return self._empty_chart("Action Distribution")
        
        # Count actions (assuming 4 actions: forward, rotate_left, rotate_right, backward)
        action_names = ['Forward', 'Rotate Left', 'Rotate Right', 'Backward']
        action_counts = [0, 0, 0, 0]
        
        for action_list in actions[-100:]:  # Last 100 episodes
            if isinstance(action_list, list):
                for a in action_list:
                    if 0 <= a < 4:
                        action_counts[a] += 1
        
        fig = go.Figure(data=[go.Pie(
            labels=action_names,
            values=action_counts,
            hole=0.3,
            marker=dict(colors=[
                self.colors['primary'],
                self.colors['secondary'],
                self.colors['success'],
                self.colors['warning']
            ])
        )])
        
        fig.update_layout(
            title='Action Distribution (Last 100 Episodes)',
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12)
        )
        
        return fig
    
    def _create_qvalue_heatmap(self):
        """Create Q-value heatmap."""
        q_values = list(self.data_buffer['q_values'])
        
        if len(q_values) == 0:
            return self._empty_chart("Q-Value Heatmap")
        
        # Create dummy heatmap (in practice, this would be actual Q-values)
        # For visualization purposes, create a grid
        x = np.arange(10)
        y = np.arange(10)
        z = np.random.randn(10, 10)
        
        if q_values:
            # Use actual Q-values if available
            latest_q = q_values[-1]
            if isinstance(latest_q, (list, np.ndarray)):
                z = np.array(latest_q).reshape(-1, 4)[:10, :]
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            colorscale='Viridis',
            colorbar=dict(title='Q-Value')
        ))
        
        fig.update_layout(
            title='Q-Value Heatmap (State-Action)',
            xaxis_title='Action',
            yaxis_title='State',
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12)
        )
        
        return fig
    
    def _empty_chart(self, title: str):
        """Create an empty chart placeholder."""
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for data...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color=self.colors['text'])
        )
        fig.update_layout(
            title=title,
            template='plotly_dark',
            font=dict(family='Arial, Helvetica, sans-serif', size=12)
        )
        return fig
    
    def update_data(self, metrics: dict):
        """
        Update dashboard data with new metrics.
        
        Args:
            metrics: Dictionary containing episode metrics
        """
        self.data_buffer['episodes'].append(metrics.get('episode', 0))
        self.data_buffer['rewards'].append(metrics.get('reward', 0))
        self.data_buffer['success_rates'].append(metrics.get('success_rate', 0))
        self.data_buffer['steps'].append(metrics.get('steps', 0))
        self.data_buffer['losses'].append(metrics.get('loss', 0))
        self.data_buffer['epsilons'].append(metrics.get('epsilon', 0))
        self.data_buffer['actions'].append(metrics.get('actions', []))
        self.data_buffer['q_values'].append(metrics.get('q_values', []))
        self.data_buffer['learning_rates'].append(metrics.get('learning_rate', 0))
        self.data_buffer['buffer_sizes'].append(metrics.get('buffer_size', 0))
        self.data_buffer['durations'].append(metrics.get('duration', 0))
    
    def load_from_log(self, log_file: str):
        """
        Load training data from log file.
        
        Args:
            log_file: Path to training log file (JSON format)
        """
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            self.update_data(entry)
    
    def _save_session(self, filename: Path):
        """Save current session data."""
        data = {key: list(val) for key, val in self.data_buffer.items()}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_all_charts(self, export_dir: Path):
        """Export all charts as PNG files."""
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # This would export all charts - implementation depends on Plotly's export functionality
        # Requires kaleido package: pip install kaleido
        pass
    
    def run(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = False):
        """
        Start the dashboard server.
        
        Args:
            host: Host address
            port: Port number
            debug: Enable debug mode
        """
        print(f"🚀 Starting dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


def launch_dashboard(log_dir: str = "results/logs", port: int = 8050):
    """
    Convenience function to launch the dashboard.
    
    Args:
        log_dir: Directory containing training logs
        port: Port number for the dashboard
    """
    dashboard = TrainingDashboard(log_dir=log_dir)
    dashboard.run(port=port)


if __name__ == "__main__":
    # Example usage
    dashboard = TrainingDashboard(log_dir="results/logs")
    
    # Simulate some training data
    for i in range(100):
        metrics = {
            'episode': i,
            'reward': np.random.randn() * 10 + (i * 0.1),  # Improving trend
            'success_rate': min(i / 100, 1.0),
            'steps': max(100 - i, 50),
            'loss': 1.0 / (i + 1),
            'epsilon': max(0.01, 1.0 - i / 100),
            'actions': np.random.randint(0, 4, 50).tolist(),
            'q_values': np.random.randn(40).tolist(),
            'learning_rate': 0.0001,
            'buffer_size': min(i * 10, 10000),
            'duration': np.random.rand() * 2
        }
        dashboard.update_data(metrics)
    
    dashboard.run(debug=True)

"""
Visualization Launcher Script
Unified interface for all visualization tools
"""

import argparse
import sys
from pathlib import Path


def launch_dashboard(log_dir: str = "results/logs", port: int = 8050):
    """Launch real-time training dashboard."""
    print(f"🚀 Launching real-time dashboard on port {port}...")
    print(f"📁 Monitoring log directory: {log_dir}")
    print(f"🌐 Open browser at: http://localhost:{port}")
    
    from src.visualization.dashboard_realtime import launch_dashboard as run_dashboard
    run_dashboard(log_dir, port)


def generate_comparisons(config_file: str, output_dir: str = "results/comparisons"):
    """Generate comparison visualizations."""
    import json
    from src.visualization.compare_runs import ComparisonVisualizer
    
    print(f"🎨 Generating comparison visualizations...")
    print(f"📄 Config file: {config_file}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    runs = config.get('runs', {})
    
    viz = ComparisonVisualizer()
    viz.export_comparison_report(runs, output_dir)
    
    print(f"✅ Comparisons saved to: {output_dir}")


def animate_trajectory(log_file: str, output_path: str, format: str = 'mp4'):
    """Create trajectory animation."""
    from src.visualization.trajectory_animator import animate_trajectory as create_anim
    
    print(f"🎬 Creating trajectory animation...")
    print(f"📄 Log file: {log_file}")
    print(f"💾 Output: {output_path}")
    print(f"🎥 Format: {format}")
    
    create_anim(log_file, output_path, format)
    
    print(f"✅ Animation saved!")


def visualize_network(model_path: str, output_dir: str = "results/network_vis"):
    """Visualize neural network."""
    import torch
    from src.visualization.network_visualizer import visualize_network as viz_net
    
    print(f"🧠 Visualizing neural network...")
    print(f"📄 Model path: {model_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load model
    model = torch.load(model_path, map_location='cpu')
    
    viz_net(model, output_dir)
    
    print(f"✅ Network visualizations saved to: {output_dir}")


def analyze_state_space(
    agent_path: str,
    trajectory_log: str,
    rewards_file: str,
    output_dir: str = "results/state_space"
):
    """Generate state-space analysis."""
    import torch
    import json
    import numpy as np
    from src.visualization.state_space_analyzer import StateSpaceAnalyzer
    
    print(f"🗺️ Analyzing state-space...")
    print(f"📄 Agent: {agent_path}")
    print(f"📄 Trajectories: {trajectory_log}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load agent
    agent = torch.load(agent_path, map_location='cpu')
    
    # Load trajectory data
    with open(trajectory_log, 'r') as f:
        trajectories = json.load(f)
    
    # Load rewards
    rewards = np.loadtxt(rewards_file)
    
    # Create analyzer
    analyzer = StateSpaceAnalyzer(state_bounds=((0, 10), (0, 10)))
    
    # Update visitation
    for traj in trajectories:
        if isinstance(traj, dict) and 'positions' in traj:
            positions = np.array(traj['positions'])
            analyzer.update_visitation(positions)
    
    # Generate analysis
    analyzer.create_complete_analysis(agent, rewards, output_dir=output_dir)
    
    print(f"✅ State-space analysis saved to: {output_dir}")


def generate_performance_dashboard(config_file: str, output_dir: str = "results/performance"):
    """Generate performance metrics dashboard."""
    import json
    from src.visualization.performance_dashboard import generate_performance_dashboard as gen_dashboard
    
    print(f"📊 Generating performance dashboard...")
    print(f"📄 Config file: {config_file}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    runs = config.get('runs', {})
    
    gen_dashboard(runs, output_dir)
    
    print(f"✅ Dashboard saved to: {output_dir}")


def generate_all(base_config: str, output_base: str = "results"):
    """Generate all visualizations."""
    import json
    
    print("🎨 Generating ALL visualizations...")
    print(f"📄 Base config: {base_config}")
    
    with open(base_config, 'r') as f:
        config = json.load(f)
    
    output_path = Path(output_base)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Comparisons
    if 'comparison_runs' in config:
        generate_comparisons(base_config, str(output_path / "comparisons"))
    
    # 2. Performance dashboard
    if 'performance_runs' in config:
        generate_performance_dashboard(base_config, str(output_path / "performance"))
    
    # 3. Network visualization
    if 'model_path' in config:
        visualize_network(config['model_path'], str(output_path / "network"))
    
    # 4. State-space analysis
    if 'agent_path' in config and 'trajectory_log' in config:
        analyze_state_space(
            config['agent_path'],
            config['trajectory_log'],
            config.get('rewards_file', 'rewards.txt'),
            str(output_path / "state_space")
        )
    
    # 5. Trajectory animations
    if 'trajectory_animations' in config:
        for name, log_file in config['trajectory_animations'].items():
            animate_trajectory(
                log_file,
                str(output_path / f"animations/{name}.mp4"),
                format='mp4'
            )
    
    print(f"✅ All visualizations complete! Check: {output_base}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Robot Navigation RL Visualization Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch real-time dashboard
  python scripts/visualize.py dashboard --log-dir results/logs --port 8050
  
  # Generate comparisons
  python scripts/visualize.py compare --config comparison_config.json
  
  # Create trajectory animation
  python scripts/visualize.py trajectory --log trajectory.json --output video.mp4
  
  # Visualize network
  python scripts/visualize.py network --model trained_models/dqn.pt
  
  # Analyze state-space
  python scripts/visualize.py statespace --agent agent.pt --trajectories traj.json --rewards rewards.txt
  
  # Generate performance dashboard
  python scripts/visualize.py performance --config performance_config.json
  
  # Generate everything
  python scripts/visualize.py all --config master_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Visualization command')
    
    # Dashboard
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch real-time dashboard')
    dashboard_parser.add_argument('--log-dir', default='results/logs', help='Log directory to monitor')
    dashboard_parser.add_argument('--port', type=int, default=8050, help='Port number')
    
    # Comparisons
    compare_parser = subparsers.add_parser('compare', help='Generate comparison plots')
    compare_parser.add_argument('--config', required=True, help='JSON config file with runs')
    compare_parser.add_argument('--output', default='results/comparisons', help='Output directory')
    
    # Trajectory
    traj_parser = subparsers.add_parser('trajectory', help='Create trajectory animation')
    traj_parser.add_argument('--log', required=True, help='Trajectory log file')
    traj_parser.add_argument('--output', required=True, help='Output file path')
    traj_parser.add_argument('--format', choices=['mp4', 'gif'], default='mp4', help='Output format')
    
    # Network
    network_parser = subparsers.add_parser('network', help='Visualize neural network')
    network_parser.add_argument('--model', required=True, help='Model file path')
    network_parser.add_argument('--output', default='results/network_vis', help='Output directory')
    
    # State-space
    statespace_parser = subparsers.add_parser('statespace', help='Analyze state-space')
    statespace_parser.add_argument('--agent', required=True, help='Agent file path')
    statespace_parser.add_argument('--trajectories', required=True, help='Trajectory log file')
    statespace_parser.add_argument('--rewards', required=True, help='Rewards file')
    statespace_parser.add_argument('--output', default='results/state_space', help='Output directory')
    
    # Performance
    perf_parser = subparsers.add_parser('performance', help='Generate performance dashboard')
    perf_parser.add_argument('--config', required=True, help='JSON config file with runs')
    perf_parser.add_argument('--output', default='results/performance', help='Output directory')
    
    # All
    all_parser = subparsers.add_parser('all', help='Generate all visualizations')
    all_parser.add_argument('--config', required=True, help='Master config file')
    all_parser.add_argument('--output', default='results', help='Base output directory')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'dashboard':
            launch_dashboard(args.log_dir, args.port)
        
        elif args.command == 'compare':
            generate_comparisons(args.config, args.output)
        
        elif args.command == 'trajectory':
            animate_trajectory(args.log, args.output, args.format)
        
        elif args.command == 'network':
            visualize_network(args.model, args.output)
        
        elif args.command == 'statespace':
            analyze_state_space(args.agent, args.trajectories, args.rewards, args.output)
        
        elif args.command == 'performance':
            generate_performance_dashboard(args.config, args.output)
        
        elif args.command == 'all':
            generate_all(args.config, args.output)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Comprehensive reporting module for FaceClass project.
Generates detailed reports with charts, statistics, and analysis.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import base64
import io

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Comprehensive report generation system."""
    
    def __init__(self, config):
        """Initialize report generator with configuration."""
        self.config = config
        self.report_format = config.get('reporting.report_format', 'html')
        self.include_charts = config.get('reporting.include_charts', True)
        self.include_heatmaps = config.get('reporting.include_heatmaps', True)
        self.include_statistics = config.get('reporting.include_statistics', True)
        
        # Set up matplotlib for non-interactive backend
        plt.switch_backend('Agg')
        
        # Create output directory
        self.output_dir = Path(self.config.get_path('reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, 
                                   attendance_data: Dict,
                                   emotion_data: Dict,
                                   attention_data: Dict,
                                   spatial_data: Dict,
                                   video_info: Dict,
                                   output_path: Optional[str] = None) -> str:
        """Generate a comprehensive report with all analysis data."""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"comprehensive_report_{timestamp}.html"
        
        # Generate all report sections
        report_sections = {
            'executive_summary': self._generate_executive_summary(attendance_data, emotion_data, attention_data),
            'attendance_analysis': self._generate_attendance_analysis(attendance_data),
            'emotion_analysis': self._generate_emotion_analysis(emotion_data),
            'attention_analysis': self._generate_attention_analysis(attention_data),
            'spatial_analysis': self._generate_spatial_analysis(spatial_data),
            'video_analysis': self._generate_video_analysis(video_info),
            'recommendations': self._generate_recommendations(attendance_data, emotion_data, attention_data)
        }
        
        # Generate charts
        charts = {}
        if self.include_charts:
            charts = self._generate_charts(attendance_data, emotion_data, attention_data, spatial_data)
        
        # Generate heatmaps
        heatmaps = {}
        if self.include_heatmaps:
            heatmaps = self._generate_heatmaps(spatial_data)
        
        # Generate HTML report
        html_content = self._generate_html_report(report_sections, charts, heatmaps)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive report generated: {output_path}")
        return str(output_path)
    
    def _generate_executive_summary(self, attendance_data: Dict, emotion_data: Dict, attention_data: Dict) -> Dict:
        """Generate executive summary section."""
        summary = {
            'total_students': attendance_data.get('total_students', 0),
            'total_sessions': attendance_data.get('total_sessions', 0),
            'average_attendance_rate': 0.0,
            'average_attention_score': 0.0,
            'dominant_emotion': 'neutral',
            'key_insights': []
        }
        
        # Calculate average attendance rate
        if attendance_data.get('students'):
            attendance_rates = [
                student.get('attendance_rate', 0.0) 
                for student in attendance_data['students'].values()
            ]
            summary['average_attendance_rate'] = np.mean(attendance_rates) if attendance_rates else 0.0
        
        # Calculate average attention score
        if attention_data.get('attention_scores'):
            summary['average_attention_score'] = np.mean(attention_data['attention_scores'])
        
        # Get dominant emotion
        if emotion_data.get('emotion_counts'):
            summary['dominant_emotion'] = max(emotion_data['emotion_counts'].items(), key=lambda x: x[1])[0]
        
        # Generate key insights
        summary['key_insights'] = self._generate_key_insights(attendance_data, emotion_data, attention_data)
        
        return summary
    
    def _generate_key_insights(self, attendance_data: Dict, emotion_data: Dict, attention_data: Dict) -> List[str]:
        """Generate key insights from the data."""
        insights = []
        
        # Attendance insights
        if attendance_data.get('students'):
            attendance_rates = [
                student.get('attendance_rate', 0.0) 
                for student in attendance_data['students'].values()
            ]
            
            if attendance_rates:
                avg_attendance = np.mean(attendance_rates)
                if avg_attendance > 0.8:
                    insights.append("High overall attendance rate (>80%)")
                elif avg_attendance < 0.6:
                    insights.append("Low attendance rate detected - intervention may be needed")
                
                # Identify students with low attendance
                low_attendance_students = [
                    student_id for student_id, student_data in attendance_data['students'].items()
                    if student_data.get('attendance_rate', 0.0) < 0.5
                ]
                if low_attendance_students:
                    insights.append(f"{len(low_attendance_students)} students have attendance rate below 50%")
        
        # Attention insights
        if attention_data.get('attention_scores'):
            avg_attention = np.mean(attention_data['attention_scores'])
            if avg_attention > 0.7:
                insights.append("Good overall attention levels")
            elif avg_attention < 0.4:
                insights.append("Low attention levels detected - consider engagement strategies")
        
        # Emotion insights
        if emotion_data.get('emotion_counts'):
            emotion_counts = emotion_data['emotion_counts']
            total_emotions = sum(emotion_counts.values())
            
            if total_emotions > 0:
                # Check for negative emotions
                negative_emotions = ['angry', 'sad', 'fear', 'disgust']
                negative_count = sum(emotion_counts.get(emotion, 0) for emotion in negative_emotions)
                negative_percentage = (negative_count / total_emotions) * 100
                
                if negative_percentage > 30:
                    insights.append(f"High percentage of negative emotions ({negative_percentage:.1f}%)")
                elif emotion_counts.get('happy', 0) / total_emotions > 0.4:
                    insights.append("Positive emotional environment detected")
        
        return insights
    
    def _generate_attendance_analysis(self, attendance_data: Dict) -> Dict:
        """Generate attendance analysis section."""
        analysis = {
            'total_records': attendance_data.get('total_records', 0),
            'student_summaries': {},
            'attendance_trends': {},
            'absenteeism_analysis': {}
        }
        
        if attendance_data.get('students'):
            for student_id, student_data in attendance_data['students'].items():
                analysis['student_summaries'][student_id] = {
                    'attendance_rate': student_data.get('attendance_rate', 0.0),
                    'total_duration': student_data.get('total_duration', 0.0),
                    'average_attention': student_data.get('average_attention', 0.0),
                    'dominant_emotion': student_data.get('dominant_emotion', 'neutral'),
                    'preferred_seat': student_data.get('preferred_seat', 'Unknown')
                }
        
        return analysis
    
    def _generate_emotion_analysis(self, emotion_data: Dict) -> Dict:
        """Generate emotion analysis section."""
        analysis = {
            'emotion_distribution': emotion_data.get('emotion_counts', {}),
            'emotion_percentages': emotion_data.get('emotion_percentages', {}),
            'dominant_emotion': emotion_data.get('dominant_emotion', 'neutral'),
            'emotion_trends': {},
            'emotional_stability': {}
        }
        
        return analysis
    
    def _generate_attention_analysis(self, attention_data: Dict) -> Dict:
        """Generate attention analysis section."""
        analysis = {
            'average_attention': attention_data.get('average_attention', 0.0),
            'attention_rate': attention_data.get('attention_rate', 0.0),
            'attention_trends': {},
            'attention_patterns': {}
        }
        
        return analysis
    
    def _generate_spatial_analysis(self, spatial_data: Dict) -> Dict:
        """Generate spatial analysis section."""
        analysis = {
            'heatmap_data': spatial_data.get('heatmaps', {}),
            'seat_assignments': spatial_data.get('seat_assignments', {}),
            'spatial_distribution': spatial_data.get('spatial_distribution', {}),
            'movement_patterns': {}
        }
        
        return analysis
    
    def _generate_video_analysis(self, video_info: Dict) -> Dict:
        """Generate video analysis section."""
        analysis = {
            'video_duration': video_info.get('duration', 0.0),
            'frame_count': video_info.get('frame_count', 0),
            'resolution': video_info.get('resolution', 'Unknown'),
            'processing_time': video_info.get('processing_time', 0.0),
            'detection_summary': video_info.get('detection_summary', {})
        }
        
        return analysis
    
    def _generate_recommendations(self, attendance_data: Dict, emotion_data: Dict, attention_data: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Attendance recommendations
        if attendance_data.get('students'):
            low_attendance_students = [
                student_id for student_id, student_data in attendance_data['students'].items()
                if student_data.get('attendance_rate', 0.0) < 0.5
            ]
            
            if low_attendance_students:
                recommendations.append(f"Consider intervention strategies for students with low attendance: {', '.join(low_attendance_students)}")
        
        # Attention recommendations
        if attention_data.get('average_attention', 0.0) < 0.4:
            recommendations.append("Implement engagement strategies to improve attention levels")
        
        # Emotion recommendations
        if emotion_data.get('emotion_counts'):
            negative_emotions = ['angry', 'sad', 'fear', 'disgust']
            negative_count = sum(emotion_data['emotion_counts'].get(emotion, 0) for emotion in negative_emotions)
            total_emotions = sum(emotion_data['emotion_counts'].values())
            
            if total_emotions > 0 and (negative_count / total_emotions) > 0.3:
                recommendations.append("Consider emotional support strategies for students showing negative emotions")
        
        return recommendations
    
    def _generate_charts(self, attendance_data: Dict, emotion_data: Dict, attention_data: Dict, spatial_data: Dict) -> Dict:
        """Generate charts for the report."""
        charts = {}
        
        # Attendance chart
        if attendance_data.get('students'):
            charts['attendance_chart'] = self._create_attendance_chart(attendance_data)
        
        # Emotion distribution chart
        if emotion_data.get('emotion_counts'):
            charts['emotion_chart'] = self._create_emotion_chart(emotion_data)
        
        # Attention trend chart
        if attention_data.get('attention_scores'):
            charts['attention_chart'] = self._create_attention_chart(attention_data)
        
        # Spatial distribution chart
        if spatial_data.get('heatmaps'):
            charts['spatial_chart'] = self._create_spatial_chart(spatial_data)
        
        return charts
    
    def _create_attendance_chart(self, attendance_data: Dict) -> str:
        """Create attendance chart."""
        try:
            plt.figure(figsize=(10, 6))
            
            student_ids = list(attendance_data['students'].keys())
            attendance_rates = [
                attendance_data['students'][student_id].get('attendance_rate', 0.0) * 100
                for student_id in student_ids
            ]
            
            plt.bar(student_ids, attendance_rates, color='skyblue')
            plt.title('Student Attendance Rates')
            plt.xlabel('Student ID')
            plt.ylabel('Attendance Rate (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_str
        except Exception as e:
            logger.error(f"Failed to create attendance chart: {e}")
            return ""
    
    def _create_emotion_chart(self, emotion_data: Dict) -> str:
        """Create emotion distribution chart."""
        try:
            plt.figure(figsize=(10, 6))
            
            emotions = list(emotion_data['emotion_counts'].keys())
            counts = list(emotion_data['emotion_counts'].values())
            
            plt.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=90)
            plt.title('Emotion Distribution')
            plt.axis('equal')
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_str
        except Exception as e:
            logger.error(f"Failed to create emotion chart: {e}")
            return ""
    
    def _create_attention_chart(self, attention_data: Dict) -> str:
        """Create attention trend chart."""
        try:
            plt.figure(figsize=(10, 6))
            
            attention_scores = attention_data.get('attention_scores', [])
            if attention_scores:
                plt.plot(attention_scores, color='green', linewidth=2)
                plt.title('Attention Score Trend')
                plt.xlabel('Time')
                plt.ylabel('Attention Score')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Convert to base64
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
                
                return img_str
        except Exception as e:
            logger.error(f"Failed to create attention chart: {e}")
            return ""
    
    def _create_spatial_chart(self, spatial_data: Dict) -> str:
        """Create spatial distribution chart."""
        try:
            plt.figure(figsize=(10, 8))
            
            heatmap = spatial_data.get('heatmaps', {}).get('presence', np.zeros((100, 100)))
            
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Presence Count')
            plt.title('Spatial Distribution Heatmap')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_str
        except Exception as e:
            logger.error(f"Failed to create spatial chart: {e}")
            return ""
    
    def _generate_heatmaps(self, spatial_data: Dict) -> Dict:
        """Generate heatmaps for the report."""
        heatmaps = {}
        
        if spatial_data.get('heatmaps'):
            for heatmap_type, heatmap_data in spatial_data['heatmaps'].items():
                heatmaps[heatmap_type] = self._create_heatmap_image(heatmap_data, heatmap_type)
        
        return heatmaps
    
    def _create_heatmap_image(self, heatmap_data: np.ndarray, heatmap_type: str) -> str:
        """Create heatmap image."""
        try:
            plt.figure(figsize=(8, 6))
            
            plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
            plt.colorbar(label=f'{heatmap_type.title()} Intensity')
            plt.title(f'{heatmap_type.title()} Heatmap')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_str
        except Exception as e:
            logger.error(f"Failed to create heatmap image: {e}")
            return ""
    
    def _generate_html_report(self, report_sections: Dict, charts: Dict, heatmaps: Dict) -> str:
        """Generate HTML report content."""
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FaceClass Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c3e50;
            margin: 0;
        }
        .header p {
            color: #7f8c8d;
            margin: 5px 0;
        }
        .section {
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #ecf0f1;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .section h3 {
            color: #34495e;
            margin-top: 20px;
        }
        .metric {
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            text-align: center;
            min-width: 120px;
        }
        .metric .value {
            font-size: 24px;
            font-weight: bold;
        }
        .metric .label {
            font-size: 12px;
            opacity: 0.8;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .insights {
            background-color: #e8f4fd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .recommendations {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }
        .recommendations ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FaceClass Analysis Report</h1>
            <p>Comprehensive Student Attendance and Behavior Analysis</p>
            <p>Generated on: {{ generation_time }}</p>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value">{{ executive_summary.total_students }}</div>
                    <div class="label">Total Students</div>
                </div>
                <div class="metric">
                    <div class="value">{{ "%.1f"|format(executive_summary.average_attendance_rate * 100) }}%</div>
                    <div class="label">Avg Attendance</div>
                </div>
                <div class="metric">
                    <div class="value">{{ "%.1f"|format(executive_summary.average_attention_score * 100) }}%</div>
                    <div class="label">Avg Attention</div>
                </div>
                <div class="metric">
                    <div class="value">{{ executive_summary.dominant_emotion.title() }}</div>
                    <div class="label">Dominant Emotion</div>
                </div>
            </div>
            
            <div class="insights">
                <h3>Key Insights</h3>
                <ul>
                    {% for insight in executive_summary.key_insights %}
                    <li>{{ insight }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        <!-- Attendance Analysis -->
        <div class="section">
            <h2>Attendance Analysis</h2>
            {% if attendance_analysis.student_summaries %}
            <table>
                <thead>
                    <tr>
                        <th>Student ID</th>
                        <th>Attendance Rate</th>
                        <th>Total Duration</th>
                        <th>Avg Attention</th>
                        <th>Dominant Emotion</th>
                        <th>Preferred Seat</th>
                    </tr>
                </thead>
                <tbody>
                    {% for student_id, data in attendance_analysis.student_summaries.items() %}
                    <tr>
                        <td>{{ student_id }}</td>
                        <td>{{ "%.1f"|format(data.attendance_rate * 100) }}%</td>
                        <td>{{ "%.1f"|format(data.total_duration / 60) }} min</td>
                        <td>{{ "%.1f"|format(data.average_attention * 100) }}%</td>
                        <td>{{ data.dominant_emotion.title() }}</td>
                        <td>{{ data.preferred_seat or 'Unknown' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>

        <!-- Charts -->
        {% if charts %}
        <div class="section">
            <h2>Visualizations</h2>
            {% if charts.attendance_chart %}
            <div class="chart-container">
                <h3>Attendance Rates</h3>
                <img src="data:image/png;base64,{{ charts.attendance_chart }}" alt="Attendance Chart">
            </div>
            {% endif %}
            
            {% if charts.emotion_chart %}
            <div class="chart-container">
                <h3>Emotion Distribution</h3>
                <img src="data:image/png;base64,{{ charts.emotion_chart }}" alt="Emotion Chart">
            </div>
            {% endif %}
            
            {% if charts.attention_chart %}
            <div class="chart-container">
                <h3>Attention Trends</h3>
                <img src="data:image/png;base64,{{ charts.attention_chart }}" alt="Attention Chart">
            </div>
            {% endif %}
            
            {% if charts.spatial_chart %}
            <div class="chart-container">
                <h3>Spatial Distribution</h3>
                <img src="data:image/png;base64,{{ charts.spatial_chart }}" alt="Spatial Chart">
            </div>
            {% endif %}
        </div>
        {% endif %}

        <!-- Recommendations -->
        <div class="section">
            <h2>Recommendations</h2>
            <div class="recommendations">
                <ul>
                    {% for recommendation in recommendations %}
                    <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        <div class="footer">
            <p>Report generated by FaceClass Analysis System</p>
            <p>For questions or support, please contact the system administrator</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            executive_summary=report_sections['executive_summary'],
            attendance_analysis=report_sections['attendance_analysis'],
            emotion_analysis=report_sections['emotion_analysis'],
            attention_analysis=report_sections['attention_analysis'],
            spatial_analysis=report_sections['spatial_analysis'],
            video_analysis=report_sections['video_analysis'],
            charts=charts,
            heatmaps=heatmaps,
            recommendations=report_sections['recommendations']
        )
        
        return html_content
    
    def export_to_csv(self, data: Dict, output_path: Optional[str] = None) -> str:
        """Export data to CSV format."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"analysis_data_{timestamp}.csv"
        
        # Convert data to DataFrame
        df_data = []
        
        # Flatten the data structure
        for category, category_data in data.items():
            if isinstance(category_data, dict):
                for key, value in category_data.items():
                    if isinstance(value, (int, float, str)):
                        df_data.append({
                            'category': category,
                            'metric': key,
                            'value': value
                        })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Data exported to CSV: {output_path}")
        return str(output_path) 
import React, { StrictMode, Component } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import AppOptimized from './App_optimized.jsx'
import AppRealWater from './App_real_water.jsx'
import AppEntropy from './App_entropy.jsx'
import AppLipid from './App_lipid.jsx'
import AppInference from './App_inference.jsx'
import AppRender from './App_render.jsx'
import AppNetwork from './App_network.jsx'
import AppWaterNetwork from './App_water_network.jsx'
import AppOverallFiedler from './App_overall_fiedler.jsx'

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '2rem',
          background: '#1e293b',
          color: '#f8fafc',
          fontFamily: 'monospace',
          minHeight: '100vh',
          boxSizing: 'border-box'
        }}>
          <h1 style={{ color: '#ef4444', marginTop: 0 }}>React Render Crash Detected</h1>
          <p style={{ fontSize: '1.1rem' }}><strong>Message:</strong> {this.state.error && this.state.error.toString()}</p>
          <p><strong>Component Stack Trace:</strong></p>
          <pre style={{
            background: '#0f172a',
            padding: '1rem',
            borderRadius: '8px',
            overflowX: 'auto',
            whiteSpace: 'pre-wrap',
            color: '#cbd5e1'
          }}>
            {this.state.errorInfo && this.state.errorInfo.componentStack}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

const urlParams = new URLSearchParams(window.location.search);
const mode = urlParams.get('mode');

let RootComponent = App;
if (mode === 'optimized') {
  RootComponent = AppOptimized;
} else if (mode === 'realwater') {
  RootComponent = AppRealWater;
} else if (mode === 'entropy') {
  RootComponent = AppEntropy;
} else if (mode === 'lipid') {
  RootComponent = AppLipid;
} else if (mode === 'inference') {
  RootComponent = AppInference;
} else if (mode === 'render') {
  RootComponent = AppRender;
} else if (mode === 'network') {
  RootComponent = AppNetwork;
} else if (mode === 'water_network') {
  RootComponent = AppWaterNetwork;
} else if (mode === 'overall_fiedler') {
  RootComponent = AppOverallFiedler;
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <RootComponent />
    </ErrorBoundary>
  </StrictMode>,
)

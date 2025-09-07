declare module 'react-plotly.js' {
  import { ComponentType } from 'react';
  import { PlotParams } from 'plotly.js';

  interface PlotProps extends Partial<PlotParams> {
    data: PlotParams['data'];
    layout?: Partial<PlotParams['layout']>;
    config?: Partial<PlotParams['config']>;
    style?: React.CSSProperties;
    className?: string;
    onInitialized?: (figure: PlotParams, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: PlotParams, graphDiv: HTMLElement) => void;
    onPurge?: (figure: PlotParams, graphDiv: HTMLElement) => void;
    onError?: (err: Error) => void;
    onSelected?: (eventData: any) => void;
    onDeselect?: () => void;
    onClick?: (eventData: any) => void;
    onHover?: (eventData: any) => void;
    onUnhover?: (eventData: any) => void;
    onRelayout?: (eventData: any) => void;
    onRestyle?: (eventData: any) => void;
    onRedraw?: () => void;
    onAnimated?: () => void;
    onAnimatingFrame?: (eventData: any) => void;
    onAnimationInterrupt?: () => void;
    onAutoSize?: () => void;
    onBeforeExport?: () => void;
    onAfterExport?: () => void;
    onAfterPlot?: () => void;
    onBeforePlot?: () => void;
    onLegendClick?: (eventData: any) => boolean;
    onLegendDoubleClick?: (eventData: any) => boolean;
    onSliderChange?: (eventData: any) => void;
    onSliderEnd?: (eventData: any) => void;
    onSliderStart?: (eventData: any) => void;
    onTransitioning?: () => void;
    onTransitionInterrupted?: () => void;
    onWebGlContextLost?: () => void;
    debug?: boolean;
    useResizeHandler?: boolean;
    divId?: string;
    revision?: number;
  }

  const Plot: ComponentType<PlotProps>;
  export default Plot;
}
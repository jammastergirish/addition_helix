declare module "react-katex" {
  import { ReactNode } from "react";
  interface MathProps {
    math: string;
    errorColor?: string;
    renderError?: (error: Error) => ReactNode;
  }
  export const InlineMath: (props: MathProps) => JSX.Element;
  export const BlockMath: (props: MathProps) => JSX.Element;
}

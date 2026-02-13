declare module 'gifenc' {
    export class GIFEncoder {
        constructor(opts?: any);
        writeFrame(index: any, width: number, height: number, opts?: any): void;
        finish(): void;
        bytes(): Uint8Array;
        bytesView(): Uint8Array;
    }
    export function quantize(data: Uint8ClampedArray | Uint8Array, maxColors: number, opts?: { format?: string, oneBitAlpha?: boolean | number, clearAlpha?: boolean, clearAlphaThreshold?: number, clearAlphaColor?: number }): number[][];
    export function applyPalette(data: Uint8ClampedArray | Uint8Array, palette: number[][], format?: string): Uint8Array;
}

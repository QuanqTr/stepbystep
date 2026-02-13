"use client"
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Upload, Layers, ChevronRight, Download, Trash2, Zap, Lasso, Wand2, MousePointer2, Undo2, Redo2, ZoomIn, ZoomOut, X, FileImage, Hand } from 'lucide-react';
import { motion } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';
import { GIFEncoder, quantize, applyPalette } from 'gifenc';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

// Types based on the backend response
interface Point {
    x: number;
    y: number;
    w?: number; // Variable width
}

interface Segment {
    id: number;
    group_id: number;
    width: number; // Avg width fallback
    points: Point[];
    bbox: { x: number; y: number; w: number; h: number };
    centroid: { x: number; y: number };
}

interface Step {
    id: number;
    segments: number[]; // Array of segment IDs
}

type ToolType = 'select' | 'hand' | 'lasso' | 'wand' | 'cluster' | 'eraser';

export default function WorkspacePage() {
    const [image, setImage] = useState<string | null>(null);
    const [originalImageFile, setOriginalImageFile] = useState<HTMLImageElement | null>(null);
    const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
    const [segments, setSegments] = useState<Segment[]>([]);

    const [history, setHistory] = useState<{ steps: Step[], currentStepId: number }[]>([]);
    const [historyIndex, setHistoryIndex] = useState(-1);

    const [steps, setSteps] = useState<Step[]>([{ id: 1, segments: [] }]);
    const [currentStepId, setCurrentStepId] = useState<number>(1);
    const [isProcessing, setIsProcessing] = useState(false);
    const [hoveredSegmentId, setHoveredSegmentId] = useState<number | null>(null);

    // Transform
    const [transform, setTransform] = useState({ scale: 1, x: 0, y: 0 });
    const [isPanning, setIsPanning] = useState(false);
    const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

    // Tools
    const [activeTool, setActiveTool] = useState<ToolType>('select');
    const [brushRadius] = useState(20);
    const [lassoPath, setLassoPath] = useState<Point[]>([]);
    const [isDragSelect, setIsDragSelect] = useState(false);
    const [showOriginal, setShowOriginal] = useState(true);
    const [isDeleteMode, setIsDeleteMode] = useState(false);
    const [downloadAll, setDownloadAll] = useState(false);
    const [exportFormat, setExportFormat] = useState<'png' | 'svg' | 'webp'>('png');

    // GIF Modal State
    const [showGifModal, setShowGifModal] = useState(false);
    const [gifBlob, setGifBlob] = useState<Blob | null>(null);
    const [gifUrl, setGifUrl] = useState<string | null>(null);
    const [gifTransparent, setGifTransparent] = useState(false);
    const [isGeneratingGif, setIsGeneratingGif] = useState(false);

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // --- History Management ---
    const pushHistory = useCallback((newSteps: Step[], newCurrentStepId: number) => {
        const newState = { steps: newSteps, currentStepId: newCurrentStepId };
        const validHistory = history.slice(0, historyIndex + 1);
        setHistory([...validHistory, newState]);
        setHistoryIndex(validHistory.length);
        setSteps(newSteps);
        setCurrentStepId(newCurrentStepId);
    }, [history, historyIndex]);

    const undo = useCallback(() => {
        if (historyIndex > 0) {
            const prevState = history[historyIndex - 1];
            setHistoryIndex(historyIndex - 1);
            setSteps(prevState.steps);
            setCurrentStepId(prevState.currentStepId);
        }
    }, [history, historyIndex]);

    const redo = useCallback(() => {
        if (historyIndex < history.length - 1) {
            const nextState = history[historyIndex + 1];
            setHistoryIndex(historyIndex + 1);
            setSteps(nextState.steps);
            setCurrentStepId(nextState.currentStepId);
        }
    }, [history, historyIndex]);

    const getAssignedSegmentIds = () => {
        const assigned = new Set<number>();
        steps.forEach(step => step.segments.forEach(id => assigned.add(id)));
        return assigned;
    };
    const assignedSegments = getAssignedSegmentIds();

    // --- Initial Setup ---
    const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const imageUrl = URL.createObjectURL(file);
        setImage(imageUrl);
        setIsProcessing(true);

        const imgObj = new Image();
        imgObj.src = imageUrl;
        imgObj.onload = () => setOriginalImageFile(imgObj);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/process-image', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error('Failed to process image');

            const data = await res.json();
            setSegments(data.segments);
            setImageSize({ width: data.width, height: data.height });

            const initialSteps = [{ id: 1, segments: [] }];
            setSteps(initialSteps); // Reset steps
            setHistory([{ steps: initialSteps, currentStepId: 1 }]);
            setHistoryIndex(0);
            setTransform({ scale: 1, x: 0, y: 0 }); // Reset zoom
        } catch (err) {
            console.error(err);
            alert('Error processing image. Please try again.');
        } finally {
            setIsProcessing(false);
        }
    };




    // --- Selection Logic ---
    const updateStepsWithSelection = useCallback((newSteps: Step[]) => {
        pushHistory(newSteps, currentStepId);
    }, [pushHistory, currentStepId]);

    const selectSegment = (segmentId: number, multi = false) => {
        const newSteps = steps.map(step => {
            if (step.id === currentStepId) {
                // Delete Mode: Always remove
                if (isDeleteMode) {
                    return { ...step, segments: step.segments.filter(id => id !== segmentId) };
                }

                // Normal Mode: Toggle
                if (step.segments.includes(segmentId) && !multi) {
                    return { ...step, segments: step.segments.filter(id => id !== segmentId) };
                }
                if (!step.segments.includes(segmentId)) {
                    return { ...step, segments: [...step.segments, segmentId] };
                }
            }
            return step;
        });
        updateStepsWithSelection(newSteps);
    };

    const selectSegments = (ids: number[]) => {
        if (ids.length === 0) return;
        const newSteps = steps.map(step => {
            if (step.id === currentStepId) {
                if (isDeleteMode) {
                    // Remove these IDs
                    return { ...step, segments: step.segments.filter(id => !ids.includes(id)) };
                } else {
                    // Add these IDs
                    const newSegments = new Set(step.segments);
                    ids.forEach(id => newSegments.add(id));
                    return { ...step, segments: Array.from(newSegments) };
                }
            }
            return step;
        });
        updateStepsWithSelection(newSteps);
    };


    const deselectAllInStep = useCallback(() => {
        const newSteps = steps.map(step => {
            if (step.id === currentStepId) {
                return { ...step, segments: [] };
            }
            return step;
        });
        updateStepsWithSelection(newSteps);
    }, [steps, currentStepId, pushHistory]); // Added callback and dependencies

    // --- Mouse & Tool Handling ---
    const getCanvasPoint = (e: React.MouseEvent<HTMLCanvasElement>): Point => {
        if (!canvasRef.current) return { x: 0, y: 0 };
        const rect = canvasRef.current.getBoundingClientRect();
        const clientX = e.clientX - rect.left;
        const clientY = e.clientY - rect.top;
        return {
            x: (clientX - transform.x) / transform.scale,
            y: (clientY - transform.y) / transform.scale
        };
    };

    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        // Pan Logic: Hand Tool OR Middle Click OR Space+Click
        if (activeTool === 'hand' || e.button === 1 || e.shiftKey) {
            setIsPanning(true);
            setLastMousePos({ x: e.clientX, y: e.clientY });
            return;
        }

        const pos = getCanvasPoint(e);

        if (activeTool === 'lasso') {
            setIsDragSelect(true);
            setLassoPath([pos]);
        }
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (isPanning) {
            const dx = e.clientX - lastMousePos.x;
            const dy = e.clientY - lastMousePos.y;
            setTransform(prev => ({ ...prev, x: prev.x + dx, y: prev.y + dy }));
            setLastMousePos({ x: e.clientX, y: e.clientY });
            return;
        }

        const pos = getCanvasPoint(e);

        if (activeTool === 'lasso' && isDragSelect) {
            setLassoPath(prev => [...prev, pos]);
            return;
        }

        if (activeTool === 'eraser' && isDragSelect) {
            // Eraser Drag Logic
            let foundId: number | null = null;
            const hitRadius = 10 / transform.scale;
            for (const seg of segments) {
                if (pos.x >= seg.bbox.x - hitRadius && pos.x <= seg.bbox.x + seg.bbox.w + hitRadius &&
                    pos.y >= seg.bbox.y - hitRadius && pos.y <= seg.bbox.y + seg.bbox.h + hitRadius) {
                    const isNear = seg.points.some(p => Math.abs(p.x - pos.x) < hitRadius && Math.abs(p.y - pos.y) < hitRadius);
                    if (isNear) {
                        // Check if selected in current step
                        const currentStep = steps.find(s => s.id === currentStepId);
                        if (currentStep && currentStep.segments.includes(seg.id)) {
                            selectSegment(seg.id); // Will trigger eraser logic
                        }
                        break;
                    }
                }
            }
            return;
        }

        if (activeTool !== 'hand' && activeTool !== 'lasso') {
            const hitRadius = 5 / transform.scale;

            for (const seg of segments) {
                // Check bbox first
                if (pos.x >= seg.bbox.x - hitRadius && pos.x <= seg.bbox.x + seg.bbox.w + hitRadius &&
                    pos.y >= seg.bbox.y - hitRadius && pos.y <= seg.bbox.y + seg.bbox.h + hitRadius) {
                    // Precise check
                    const isNear = seg.points.some(p => Math.abs(p.x - pos.x) < hitRadius && Math.abs(p.y - pos.y) < hitRadius);
                    if (isNear) {
                        foundId = seg.id;
                        break;
                    }
                }
            }
            setHoveredSegmentId(foundId);
        }
    };

    const handleMouseUp = () => {
        setIsPanning(false);
        if (activeTool === 'lasso' && isDragSelect) {
            setIsDragSelect(false);
            if (lassoPath.length > 2) {
                const selectedIds = segments.filter(seg => isPointInPolygon(seg.centroid, lassoPath)).map(s => s.id);
                selectSegments(selectedIds);
            }
            setLassoPath([]);
        }
    };

    const handleWheel = (e: React.WheelEvent) => {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault(); // Browser zoom
        }
        // Simple zoom on scroll
        const zoomSensitivity = 0.001;
        const delta = -e.deltaY * zoomSensitivity;
        const newScale = Math.min(Math.max(0.1, transform.scale + delta), 5);
        setTransform(prev => ({ ...prev, scale: newScale }));
    };

    const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (activeTool === 'lasso' || isPanning || activeTool === 'hand') return;

        const pos = getCanvasPoint(e);

        if (activeTool === 'wand') {
            const localRadius = brushRadius;
            const selectedIds = segments.filter(seg => {
                const dx = seg.centroid.x - pos.x;
                const dy = seg.centroid.y - pos.y;
                return Math.sqrt(dx * dx + dy * dy) <= localRadius;
            }).map(s => s.id);

            selectSegments(selectedIds);
            return;
        }

        if (activeTool === 'cluster' && hoveredSegmentId) {
            const seg = segments.find(s => s.id === hoveredSegmentId);
            if (seg && seg.group_id !== -1) {
                const groupIds = segments.filter(s => s.group_id === seg.group_id).map(s => s.id);
                selectSegments(groupIds);
            } else if (seg) {
                selectSegment(seg.id);
            }
            return;
        }

        if (hoveredSegmentId) {
            selectSegment(hoveredSegmentId);
        }
    };

    const isPointInPolygon = (p: Point, polygon: Point[]) => {
        let isInside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            if ((polygon[i].y > p.y) !== (polygon[j].y > p.y) &&
                p.x < (polygon[j].x - polygon[i].x) * (p.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x) {
                isInside = !isInside;
            }
        }
        return isInside;
    };

    const autoDistribute = () => {
        if (segments.length === 0) return;
        const sortedSegments = [...segments].sort((a, b) => a.bbox.y - b.bbox.y);
        const numSteps = 5;
        const chunkSize = Math.ceil(sortedSegments.length / numSteps);

        const newSteps: Step[] = [];
        for (let i = 0; i < numSteps; i++) {
            const chunk = sortedSegments.slice(i * chunkSize, (i + 1) * chunkSize);
            newSteps.push({
                id: i + 1,
                segments: chunk.map(s => s.id)
            });
        }
        pushHistory(newSteps, 1);
    };

    // --- Export Logic ---
    const drawSegmentsOnCanvas = (ctx: CanvasRenderingContext2D, stepId: number, includePrevious: boolean) => {
        // Helper to draw specific state
        const stepIndex = steps.findIndex(s => s.id === stepId);
        if (stepIndex === -1) return;

        // User requested: "ảnh tải về ở step này thì không trùng với step trước"
        // This means EXCLUSIVE export (only segments in current step).
        // Ignoring 'includePrevious' param for now based on user request, or setting it to false.

        // const relevantSteps = includePrevious ? steps.slice(0, stepIndex + 1) : [steps[stepIndex]];
        const relevantSteps = [steps[stepIndex]]; // EXCLUSIVE

        relevantSteps.forEach(step => {
            step.segments.forEach(segId => {
                const seg = segments.find(s => s.id === segId);
                if (seg) drawSegmentPath(ctx, seg, '#000000', 1.0);
            });
        });
    };

    const exportImage = (stepId?: number) => {
        if (!imageSize.width) return;
        const canvas = document.createElement('canvas');
        canvas.width = imageSize.width;
        canvas.height = imageSize.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Default to current step (Cumulative converted to Exclusive)
        const targetStepId = stepId || currentStepId;
        drawSegmentsOnCanvas(ctx, targetStepId, false); // False = Exclusive

        canvas.toBlob(blob => {
            if (blob) saveAs(blob, `step-${targetStepId}-${Date.now()}.png`);
        });
    };

    const exportAllZIP = async () => {
        if (steps.length === 0) return;
        const zip = new JSZip();
        const folder = zip.folder("steps");

        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            const canvas = document.createElement('canvas');
            canvas.width = imageSize.width;
            canvas.height = imageSize.height;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                // Draw cumulative up to this step
                const relevantSteps = [steps[i]]; // EXCLUSIVE
                relevantSteps.forEach(s => {
                    s.segments.forEach(segId => {
                        const seg = segments.find(sg => sg.id === segId);
                        if (seg) drawSegmentPath(ctx, seg, '#000000', 1.0);
                    });
                });

                const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve));
                if (blob) {
                    folder?.file(`step-${i + 1}.png`, blob);
                }
            }
        }

        const content = await zip.generateAsync({ type: "blob" });
        saveAs(content, `drawing-steps-${Date.now()}.zip`);
    };

    const exportSVG = (stepId?: number) => {
        if (segments.length === 0) return;
        let svgContent = `<svg width="${imageSize.width}" height="${imageSize.height}" xmlns="http://www.w3.org/2000/svg">
        <style>path { fill: none !important; stroke: black; stroke-linecap: round; stroke-linejoin: round; }</style>
        <g fill="none" stroke="black" stroke-width="1">`;

        // Export EXCLUSIVE (Current Step Only)
        const targetStepId = stepId || currentStepId;
        const stepIndex = steps.findIndex(s => s.id === targetStepId);
        const relevantSteps = (stepIndex !== -1) ? [steps[stepIndex]] : [];

        relevantSteps.forEach(step => {
            step.segments.forEach(segId => {
                const seg = segments.find(s => s.id === segId);
                if (seg) {
                    const d = seg.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
                    svgContent += `<path d="${d}" stroke="black" stroke-width="${seg.width || 1}" fill="none" fill-opacity="0" style="fill:none" stroke-linecap="round" stroke-linejoin="round" />`;
                }
            });
        });
        svgContent += '</g></svg>';
        const blob = new Blob([svgContent], { type: 'image/svg+xml' });
        saveAs(blob, `step-${targetStepId}-${Date.now()}.svg`);
    };

    const exportAllSVG = async () => {
        if (steps.length === 0) return;
        const zip = new JSZip();
        const folder = zip.folder("steps_svg");

        steps.forEach((step, index) => {
            // Generate SVG for step i (EXCLUSIVE)
            let svgContent = `<svg width="${imageSize.width}" height="${imageSize.height}" xmlns="http://www.w3.org/2000/svg">
            <style>path { fill: none !important; stroke: black; stroke-linecap: round; stroke-linejoin: round; }</style>
            <g fill="none" stroke="black" stroke-width="1">`;
            const relevantSteps = [steps[index]]; // EXCLUSIVE

            relevantSteps.forEach(s => {
                s.segments.forEach(segId => {
                    const seg = segments.find(sg => sg.id === segId);
                    if (seg) {
                        const d = seg.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
                        svgContent += `<path d="${d}" stroke="black" stroke-width="${seg.width || 1}" fill="none" fill-opacity="0" style="fill:none" stroke-linecap="round" stroke-linejoin="round" />`;
                    }
                });
            });
            svgContent += '</g></svg>';
            folder?.file(`step-${index + 1}.svg`, svgContent);
        });

        const content = await zip.generateAsync({ type: "blob" });
        saveAs(content, `drawing-steps-svg-${Date.now()}.zip`);
    };

    const exportWebP = (stepId?: number) => {
        if (!imageSize.width) return;
        const canvas = document.createElement('canvas');
        canvas.width = imageSize.width;
        canvas.height = imageSize.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const targetStepId = stepId || currentStepId;
        drawSegmentsOnCanvas(ctx, targetStepId, false);

        canvas.toBlob(blob => {
            if (blob) saveAs(blob, `step-${targetStepId}-${Date.now()}.webp`);
        }, 'image/webp');
    };

    const exportAllWebP = async () => {
        if (steps.length === 0) return;
        const zip = new JSZip();
        const folder = zip.folder("steps_webp");

        for (let i = 0; i < steps.length; i++) {
            const canvas = document.createElement('canvas');
            canvas.width = imageSize.width;
            canvas.height = imageSize.height;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                const relevantSteps = [steps[i]];
                relevantSteps.forEach(s => {
                    s.segments.forEach(segId => {
                        const seg = segments.find(sg => sg.id === segId);
                        if (seg) drawSegmentPath(ctx, seg, '#000000', 1.0);
                    });
                });

                const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(blob => resolve(blob), 'image/webp'));
                if (blob) {
                    folder?.file(`step-${i + 1}.webp`, blob);
                }
            }
        }

        const content = await zip.generateAsync({ type: "blob" });
        saveAs(content, `drawing-steps-webp-${Date.now()}.zip`);
    };

    const handleExport = () => {
        if (downloadAll) {
            switch (exportFormat) {
                case 'png': exportAllZIP(); break;
                case 'svg': exportAllSVG(); break;
                case 'webp': exportAllWebP(); break;
            }
        } else {
            switch (exportFormat) {
                case 'png': exportImage(); break;
                case 'svg': exportSVG(); break;
                case 'webp': exportWebP(); break;
            }
        }
    };

    const generateGIF = async (transparent: boolean) => {
        if (!imageSize.width || steps.length === 0) return;
        setIsGeneratingGif(true);
        setGifBlob(null); // Clear previous

        // Delay to allow UI update
        await new Promise(resolve => setTimeout(resolve, 100));

        try {
            const gif = new GIFEncoder();

            // 1. Persistent Canvas (Accumulator)
            const canvas = document.createElement('canvas');
            canvas.width = imageSize.width;
            canvas.height = imageSize.height;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            if (!ctx) return;

            // 2. Initial Setup (Blank Frame)
            // If transparent, we start with cleared canvas.
            // If white, we fill white.
            if (!transparent) {
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            }

            // Capture Blank Frame
            const { data: bData, width: bWidth, height: bHeight } = ctx.getImageData(0, 0, canvas.width, canvas.height);
            // Use 'rgba4444' if transparency is needed to ensure alpha channel is preserved in palette
            const palFormat = transparent ? 'rgba4444' : 'rgb565';
            const bPalette = quantize(bData, 256, { format: palFormat });
            const bIndex = applyPalette(bData, bPalette, palFormat);

            let bTransparentIndex = 0;
            if (transparent) {
                for (let j = 0; j < bPalette.length; j++) {
                    if (bPalette[j][3] === 0) {
                        bTransparentIndex = j;
                        break;
                    }
                }
            }
            gif.writeFrame(bIndex, bWidth, bHeight, { palette: bPalette, delay: 500, transparent: transparent, transparentIndex: bTransparentIndex });

            // 3. Incrementally Draw Steps
            for (let i = 0; i < steps.length; i++) {
                // Optimization: blend current step onto existing canvas state
                // No need to clear and redraw previous steps!

                const step = steps[i];
                // Draw ONLY current step segments
                step.segments.forEach(segId => {
                    const seg = segments.find(s => s.id === segId);
                    if (seg) drawSegmentPath(ctx, seg, '#000000', 1.0);
                });

                // Get Data
                const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);

                // Quantize & Apply Palette
                // Use 'rgba4444' for quantization to preserve alpha info if transparent
                const palette = quantize(data, 256, { format: palFormat });
                const index = applyPalette(data, palette, palFormat);

                // Find Transparent Index if needed
                let transparentIndex = 0;
                if (transparent) {
                    for (let j = 0; j < palette.length; j++) {
                        // Check Alpha channel (index 3)
                        if (palette[j][3] === 0) {
                            transparentIndex = j;
                            break;
                        }
                    }
                }

                // Write Frame
                gif.writeFrame(index, width, height, {
                    palette,
                    delay: 500,
                    transparent: transparent,
                    transparentIndex: transparentIndex
                });

                // Yield to UI thread occasionally
                if (i % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0));
            }

            gif.finish();

            // Cast to avoid TS error
            const bytes: any = gif.bytes();
            const blob = new Blob([bytes], { type: 'image/gif' });
            setGifBlob(blob);
            setGifUrl(URL.createObjectURL(blob));

        } catch (err) {
            console.error(err);
            alert("Failed to create GIF");
        } finally {
            setIsGeneratingGif(false);
        }
    };

    const openGifModal = () => {
        setShowGifModal(true);
        setGifTransparent(false); // Default to white BG
        generateGIF(false);
    };

    // Re-generate when transparent option changes (only if modal is open)
    useEffect(() => {
        if (showGifModal) {
            generateGIF(gifTransparent);
        }
    }, [gifTransparent, showGifModal]); // Added showGifModal to dependencies

    const downloadGif = () => {
        if (gifBlob) {
            saveAs(gifBlob, `drawing-process-${Date.now()}.gif`);
            setShowGifModal(false);
        }
    };

    // --- Rendering ---
    const drawSegmentPath = (ctx: CanvasRenderingContext2D, seg: Segment, color: string, alpha: number) => {
        if (seg.points.length === 0) return;
        ctx.fillStyle = color; // For circles
        ctx.strokeStyle = color; // For path fallback
        ctx.globalAlpha = alpha;

        // High Quality rendering: Draw circle at every point with specific radius
        // This handles variable width directly.
        // Performance warning: Many calls. But for < 10k points it should be fine.
        // Optimize: checks distance between points?
        ctx.beginPath();
        for (const p of seg.points) {
            const radius = (p.w && p.w > 0 ? p.w : (seg.width || 1)) / 2;
            // Draw circle
            ctx.moveTo(p.x + radius, p.y);
            ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        }
        ctx.fill();
        // Note: "Path" approach above draws disconnected circles if points are far.
        // Since these come from skeletonize (connected), they should be dense (1px neighbors).
        // If gaps appear, we might need to fill gaps.
    };

    // --- Keyboard Shortcuts ---
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

            if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
                e.preventDefault();
                undo();
                return;
            }

            switch (e.key.toLowerCase()) {
                case 'v': setActiveTool('select'); break;
                case 'h': setActiveTool('hand'); break;
                case 'r': setActiveTool('lasso'); break;
                case 't': setActiveTool('wand'); break;
                case 't': setActiveTool('wand'); break;
                case 'e': setIsDeleteMode(prev => !prev); break; // Toggle Delete Mode
                case 'delete':
                case 'backspace':
                    deselectAllInStep();
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [undo, deselectAllInStep]); // Dependencies for shortcuts

    useEffect(() => {
        if (!canvasRef.current || imageSize.width === 0) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.save();
        ctx.translate(transform.x, transform.y);
        ctx.scale(transform.scale, transform.scale);

        // Draw base image (faded)
        if (showOriginal && originalImageFile) {
            ctx.globalAlpha = 0.15;
            ctx.drawImage(originalImageFile, 0, 0, imageSize.width, imageSize.height);
            ctx.globalAlpha = 1.0;
        }

        // Draw Segments
        segments.forEach(seg => {
            const isAssigned = assignedSegments.has(seg.id);
            const isHovered = seg.id === hoveredSegmentId;
            const currentStep = steps.find(s => s.id === currentStepId);
            const isSelectedStart = currentStep?.segments.includes(seg.id);

            // Unassigned - Faint Gray
            let color = '#d1d5db';
            let alpha = 0.0; // Hide unassigned if desired, or show faint
            if (!isAssigned) alpha = 0.3;

            // Previous Steps - Black
            steps.forEach(step => {
                if (step.id < currentStepId && step.segments.includes(seg.id)) {
                    color = '#000000';
                    alpha = 1.0;
                }
            });

            // Current Step - Blue
            if (isSelectedStart) {
                color = '#3b82f6';
                alpha = 1.0;
            }

            // Hover
            if (isHovered && activeTool !== 'hand' && activeTool !== 'lasso') {
                if (activeTool === 'cluster') color = '#f59e0b';
                else color = '#60a5fa';
                alpha = 0.8;
            }

            if (alpha > 0) drawSegmentPath(ctx, seg, color, alpha);
        });

        // Draw Lasso
        if (lassoPath.length > 0) {
            ctx.beginPath();
            // High Contrast Lasso
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2 / transform.scale;
            ctx.setLineDash([5 / transform.scale, 5 / transform.scale]);
            ctx.moveTo(lassoPath[0].x, lassoPath[0].y);
            lassoPath.forEach(p => ctx.lineTo(p.x, p.y));
            ctx.stroke();

            ctx.strokeStyle = '#000000';
            ctx.setLineDash([]); // Solid outline shadow
            ctx.lineWidth = 1 / transform.scale;
            ctx.stroke(); // Double stroke for visibility

            // Semi-transparent fill
            ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
            ctx.fill();
        }

        ctx.restore();

    }, [originalImageFile, segments, steps, currentStepId, hoveredSegmentId, imageSize, lassoPath, transform, showOriginal]);

    return (
        <div className="flex h-screen bg-neutral-950 text-white overflow-hidden font-sans">
            {/* Sidebar - Steps */}
            <div className="w-80 bg-neutral-900 border-r border-neutral-800 flex flex-col z-20 shadow-xl">
                <div className="p-4 border-b border-neutral-800 bg-neutral-900 flex justify-between items-center">
                    <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-500">
                        Draw Step-by-Step
                    </h1>
                </div>

                {/* Undo/Redo & Visibility */}
                <div className="flex items-center gap-2 p-2 px-4 border-b border-neutral-800 bg-neutral-800/30 font-medium">
                    <button onClick={undo} disabled={historyIndex <= 0} className="p-1 rounded text-neutral-400 hover:text-white disabled:opacity-30">
                        <Undo2 size={18} />
                    </button>
                    <button onClick={redo} disabled={historyIndex >= history.length - 1} className="p-1 rounded text-neutral-400 hover:text-white disabled:opacity-30">
                        <Redo2 size={18} />
                    </button>
                    <div className="h-4 w-px bg-neutral-700 mx-1" />
                    <label className="flex items-center gap-2 text-xs text-neutral-400 cursor-pointer select-none">
                        <input type="checkbox" checked={showOriginal} onChange={e => setShowOriginal(e.target.checked)} className="rounded bg-neutral-800 border-neutral-700 accent-blue-500" />
                        Show Overlay
                    </label>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                    {steps.map((step, index) => (
                        <motion.div
                            layout
                            key={step.id}
                            onClick={() => setCurrentStepId(step.id)}
                            className={cn(
                                "p-3 rounded-xl border transition-all cursor-pointer group relative overflow-hidden",
                                currentStepId === step.id
                                    ? "bg-blue-900/20 border-blue-500/50 shadow-[0_0_15px_rgba(59,130,246,0.1)]"
                                    : "bg-neutral-800/50 border-neutral-700 hover:border-neutral-600"
                            )}
                        >
                            <div className="flex justify-between items-center mb-2 relative z-10">
                                <span className="font-medium text-sm text-neutral-300 group-hover:text-white">
                                    Step {index + 1}
                                </span>
                                <span className="text-xs bg-black/40 px-2 py-1 rounded text-neutral-400 font-mono">
                                    {step.segments.length} strokes
                                </span>
                            </div>

                            <div className="h-1 w-full bg-neutral-950 rounded-full overflow-hidden relative z-10">
                                <div
                                    className="h-full bg-blue-500 transition-all duration-500"
                                    style={{ width: `${Math.min(100, (step.segments.length / Math.max(1, segments.length / steps.length)) * 100)}%` }}
                                />
                            </div>
                        </motion.div>
                    ))}

                    <button
                        onClick={() => {
                            const newSteps = [...steps, { id: Date.now(), segments: [] }];
                            pushHistory(newSteps, newSteps[newSteps.length - 1].id);
                        }}
                        className="w-full py-3 border-2 border-dashed border-neutral-800 rounded-xl text-neutral-500 hover:border-neutral-700 hover:text-neutral-400 transition-colors flex items-center justify-center gap-2 hover:bg-neutral-800/50"
                    >
                        <Layers size={16} /> Add Step
                    </button>
                </div>

                <div className="p-4 border-t border-neutral-800 space-y-2 bg-neutral-900">
                    <div className="flex gap-2">
                        <button
                            onClick={autoDistribute}
                            className="flex-1 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white py-2 rounded-lg font-medium transition-colors text-sm shadow-indigo-900/20 shadow-lg"
                        >
                            <Zap size={14} /> Auto Distribute
                        </button>
                        <button onClick={deselectAllInStep} className="px-3 bg-neutral-800 hover:bg-red-900/30 hover:text-red-400 text-neutral-400 rounded-lg transition-colors">
                            <X size={16} />
                        </button>
                    </div>

                    <div className="flex flex-col gap-2">
                        <label className="flex items-center gap-2 text-xs text-neutral-400 cursor-pointer select-none bg-neutral-800/50 p-2 rounded-lg border border-neutral-800 hover:border-neutral-700 transition-colors">
                            <input
                                type="checkbox"
                                checked={downloadAll}
                                onChange={e => setDownloadAll(e.target.checked)}
                                className="rounded bg-neutral-900 border-neutral-700 accent-blue-500 w-4 h-4"
                            />
                            <span className={downloadAll ? "text-blue-400 font-medium" : ""}>Capture all steps</span>
                        </label>

                        <div className="flex gap-2">
                            <button
                                onClick={handleExport}
                                className={cn(
                                    "flex-1 flex items-center justify-center gap-2 text-white py-2 rounded-lg font-medium transition-colors text-sm",
                                    "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 shadow-lg shadow-blue-900/20"
                                )}
                            >
                                <Download size={16} />
                                Download {exportFormat.toUpperCase()} {downloadAll ? "(ZIP)" : ""}
                            </button>

                            <div className="relative">
                                <select
                                    value={exportFormat}
                                    onChange={e => setExportFormat(e.target.value as any)}
                                    className="appearance-none bg-neutral-800 text-white text-sm rounded-lg pl-3 pr-8 py-2 border border-neutral-700 focus:outline-none focus:border-blue-500 h-full cursor-pointer hover:bg-neutral-700 transition-colors"
                                >
                                    <option value="png">PNG</option>
                                    <option value="svg">SVG</option>
                                    <option value="webp">WebP</option>
                                </select>
                                <div className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-neutral-400">
                                    <ChevronRight size={14} className="rotate-90" />
                                </div>
                            </div>
                        </div>

                        <button
                            onClick={openGifModal}
                            className="w-full bg-indigo-600 hover:bg-indigo-500 text-white py-2 rounded-lg font-medium transition-colors text-sm flex items-center justify-center gap-2 shadow-lg shadow-indigo-900/20"
                            title="Generate a GIF showing step-by-step progress"
                        >
                            <FileImage size={16} /> Download GIF (Process)
                        </button>
                    </div>
                </div>

                {/* GIF Preview Modal */}
                {showGifModal && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
                        <div className="bg-neutral-900 border border-neutral-700 rounded-2xl shadow-2xl max-w-lg w-full flex flex-col overflow-hidden">
                            <div className="p-4 border-b border-neutral-800 flex justify-between items-center bg-neutral-800/50">
                                <h3 className="font-semibold text-white flex items-center gap-2">
                                    <FileImage size={18} className="text-indigo-400" />
                                    GIF Preview
                                </h3>
                                <button onClick={() => setShowGifModal(false)} className="text-neutral-400 hover:text-white transition-colors">
                                    <X size={20} />
                                </button>
                            </div>

                            <div className="p-6 flex flex-col items-center gap-4 bg-neutral-900/50">
                                <div className="relative rounded-lg overflow-hidden border border-neutral-700 bg-[url('https://media.istockphoto.com/id/1145618475/vector/checkered-flag-chequered-flag-racing-flag-vector-background.jpg?s=612x612&w=0&k=20&c=N5-802n4vVp_XGjXvL_9Qv_0_0_0_0_0.jpg')] bg-contain">
                                    {/* Checkerboard pattern simulation for transparency */}
                                    <div className="absolute inset-0 bg-neutral-800" style={{ backgroundImage: 'linear-gradient(45deg, #333 25%, transparent 25%), linear-gradient(-45deg, #333 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #333 75%), linear-gradient(-45deg, transparent 75%, #333 75%)', backgroundSize: '20px 20px', backgroundPosition: '0 0, 0 10px, 10px -10px, -10px 0px', opacity: 0.2, zIndex: 0 }} />

                                    {isGeneratingGif ? (
                                        <div className="flex flex-col items-center justify-center h-48 w-64 z-10 relative text-neutral-400 gap-2">
                                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                                            <span className="text-xs">Generating GIF...</span>
                                        </div>
                                    ) : (
                                        gifUrl && <img src={gifUrl} alt="GIF Preview" className="max-h-[300px] max-w-full object-contain z-10 relative" />
                                    )}
                                </div>

                                <label className="flex items-center gap-2 cursor-pointer select-none bg-neutral-800 hover:bg-neutral-700 p-2 px-3 rounded-lg transition-colors border border-neutral-700">
                                    <input
                                        type="checkbox"
                                        checked={gifTransparent}
                                        onChange={e => setGifTransparent(e.target.checked)}
                                        className="rounded bg-neutral-900 border-neutral-600 accent-indigo-500 w-4 h-4"
                                    />
                                    <span className="text-sm text-neutral-300">Transparent Background</span>
                                </label>
                            </div>

                            <div className="p-4 border-t border-neutral-800 flex gap-3 justify-end bg-neutral-800/30">
                                <button
                                    onClick={() => setShowGifModal(false)}
                                    className="px-4 py-2 rounded-lg text-sm font-medium text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={downloadGif}
                                    disabled={!gifBlob || isGeneratingGif}
                                    className="px-4 py-2 rounded-lg text-sm font-medium bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-indigo-900/20 flex items-center gap-2"
                                >
                                    <Download size={16} /> Download
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Main Workspace */}
            <div className="flex-1 flex flex-col relative bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-neutral-900 to-neutral-950">

                {/* Toolbar */}
                <div className="h-16 border-b border-neutral-800 flex items-center px-6 justify-between bg-neutral-900/80 backdrop-blur-md z-10 w-full">
                    <div className="flex items-center gap-2 bg-neutral-800/50 p-1 rounded-xl border border-neutral-700/50">
                        <ButtonIcon
                            active={activeTool === 'select'}
                            onClick={() => setActiveTool('select')}
                            icon={<MousePointer2 size={18} />}
                            label={isDeleteMode ? "Deselect(v)" : "Select(v)"}
                            mode={isDeleteMode ? 'delete' : 'normal'}
                        />
                        <ButtonIcon
                            active={activeTool === 'lasso'}
                            onClick={() => setActiveTool('lasso')}
                            icon={<Lasso size={18} />}
                            label={isDeleteMode ? "Erase Lasso(r)" : "Lasso(r)"}
                            mode={isDeleteMode ? 'delete' : 'normal'}
                        />
                        <ButtonIcon
                            active={activeTool === 'wand'}
                            onClick={() => setActiveTool('wand')}
                            icon={<Wand2 size={18} />}
                            label={isDeleteMode ? "Erase Wand(t)" : "Wand(t)"}
                            mode={isDeleteMode ? 'delete' : 'normal'}
                        />
                        <ButtonIcon
                            active={activeTool === 'cluster'}
                            onClick={() => setActiveTool('cluster')}
                            icon={<Layers size={18} />}
                            label={isDeleteMode ? "Erase Cluster(c)" : "Cluster(c)"}
                            mode={isDeleteMode ? 'delete' : 'normal'}
                        />

                        <div className="w-px h-6 bg-neutral-700 mx-1" />

                        <ButtonIcon
                            active={isDeleteMode}
                            onClick={() => setIsDeleteMode(!isDeleteMode)}
                            icon={<Trash2 size={18} />}
                            label="Destructive(e)"
                            className={isDeleteMode ? "bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30" : ""}
                        />
                        <ButtonIcon
                            active={activeTool === 'hand'}
                            onClick={() => setActiveTool('hand')}
                            icon={<Hand size={18} />}
                            label="Hand(h)"
                        />
                    </div>

                    <div className="flex items-center gap-2 bg-neutral-800/50 p-1 rounded-xl border border-neutral-700/50">
                        <button onClick={() => setTransform(t => ({ ...t, scale: t.scale * 1.1 }))} className="p-2 text-neutral-400 hover:text-white">
                            <ZoomIn size={18} />
                        </button>
                        <span className="text-xs w-10 text-center text-neutral-500 font-mono">{Math.round(transform.scale * 100)}%</span>
                        <button onClick={() => setTransform(t => ({ ...t, scale: t.scale / 1.1 }))} className="p-2 text-neutral-400 hover:text-white">
                            <ZoomOut size={18} />
                        </button>
                    </div>

                    <div className="flex items-center gap-2">
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleImageUpload}
                            hidden
                            accept="image/*"
                        />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-all shadow-lg shadow-blue-900/20"
                        >
                            <Upload size={16} /> Upload Image
                        </button>
                    </div>
                </div>

                {/* Canvas Area */}
                <div className="flex-1 overflow-hidden relative select-none bg-[#0a0a0a]">
                    {image ? (
                        <canvas
                            ref={canvasRef}
                            width={imageSize.width}
                            height={imageSize.height}
                            onMouseMove={handleMouseMove}
                            onMouseDown={handleMouseDown}
                            onMouseUp={handleMouseUp}
                            onWheel={handleWheel}
                            onClick={handleClick}
                            className={cn(
                                "block absolute top-0 left-0 origin-top-left",
                                activeTool === 'lasso' ? 'cursor-crosshair' :
                                    activeTool === 'wand' ? 'cursor-cell' :
                                        activeTool === 'hand' || isPanning ? 'cursor-grab active:cursor-grabbing' : 'cursor-default'
                            )}
                        />
                    ) : (
                        <div className="w-full h-full flex flex-col items-center justify-center text-neutral-600 gap-4">
                            <div className="w-24 h-24 rounded-3xl bg-neutral-900 border border-neutral-800 flex items-center justify-center shadow-inner">
                                <Upload size={40} className="text-neutral-700" />
                            </div>
                            <p>Upload a line art image to start.</p>
                        </div>
                    )}

                    {/* Zoom Hint Overlay */}
                    <div className="absolute bottom-6 right-6 text-xs text-neutral-600 flex items-center gap-4 pointer-events-none select-none">
                        <span className="bg-black/50 px-2 py-1 rounded backdrop-blur">Scroll to Zoom</span>
                        <span className="bg-black/50 px-2 py-1 rounded backdrop-blur">Shift + Drag to Pan</span>
                    </div>

                    {isProcessing && (
                        <div className="absolute inset-0 bg-black/60 flex items-center justify-center backdrop-blur-sm z-50">
                            <div className="flex flex-col items-center gap-2">
                                <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                                <span className="text-blue-400 font-mono text-sm">Processing Neural Edges...</span>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function ButtonIcon({ active, onClick, icon, label, mode = 'normal', className }: { active?: boolean, onClick: () => void, icon: React.ReactNode, label: string, mode?: 'normal' | 'delete', className?: string }) {
    return (
        <button
            onClick={onClick}
            className={cn(
                "p-2 px-3 rounded-lg transition-all flex items-center gap-2 text-sm font-medium",
                active
                    ? mode === 'delete'
                        ? "bg-red-600 text-white shadow-lg shadow-red-900/40"
                        : "bg-blue-500 text-white shadow-lg shadow-blue-900/40"
                    : "text-neutral-400 hover:bg-neutral-700 hover:text-white",
                className
            )}
        >
            {icon}
            <span className="hidden xl:inline">{label}</span>
        </button>
    )
}

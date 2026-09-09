from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.http import require_POST
import pandas as pd
import time

from .forms import CSVUploadForm
from .models import Dataset
from .datamanager import (
    upload_parsing,
    get_dataset_summary,
    generate_distribution_plot,
    generate_scatter_plot,
    reset_dataset, 
    drop_missing_rows, 
    impute_missing_values,
    filter_out_of_distribution
)


def workspace_view(request):
    """ Main wrapper page that loads the tab bar shell """
    form = CSVUploadForm()
    return render(request, 'project1/workspace.html', {'form': form})


def upload_csv(request):
    """ View called when uploading a CSV file """
    error = None
    numeric_cols, non_numeric_cols, averages, class_counts = None, None, None, None

    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']

            dataset = Dataset.objects.create(original_file=uploaded_file)
            numeric_cols, non_numeric_cols, averages, class_counts, error = upload_parsing(dataset)

            if not error:
                request.session['active_dataset_id'] = str(dataset.id)
    else:
        form = CSVUploadForm()

    context = {
        'form': form,
        'numeric_columns': numeric_cols,
        'non_numeric_columns': non_numeric_cols,
        'averages': averages,
        'class_counts': class_counts,
        'error': error
    }

    # If request is HTMX, return only the upload snippet
    if request.headers.get('HX-Request'):
        return render(request, 'project1/partials/upload_partial.html', context)
    
    return render(request, 'project1/workspace.html', context)


def explore_view(request):
    dataset_id = request.session.get('active_dataset_id')

    if not dataset_id:
        return render(request, 'project1/partials/explore_partial.html', {'dataset_id': None})

    try:
        dataset = Dataset.objects.get(id=dataset_id)
        df = pd.read_parquet(dataset.working_file_path)

        # 1. Base Summary Data
        summary_data = get_dataset_summary(df)
        columns = df.columns.tolist()

        # Check request.POST fallback to preserve selections after filtering
        params = request.POST if request.method == 'POST' else request.GET

        # 2. Handle Distribution Plot Selection
        selected_dist_col = params.get('dist_col', columns[0] if columns else None)
        dist_plot_url = generate_distribution_plot(df, dataset_id, selected_dist_col) if selected_dist_col else None

        # 3. Handle 2D Scatter Plot Selections
        x_col = params.get('x_col', columns[0] if columns else None)
        y_col = params.get('y_col', columns[1] if len(columns) > 1 else (columns[0] if columns else None))
        hue_col = params.get('hue_col', '')

        scatter_plot_url = generate_scatter_plot(df, dataset_id, x_col, y_col, hue_col) if x_col and y_col else None

        # Cache-busting timestamp for freshly regenerated images
        cache_key = f"?t={int(time.time())}"
        if dist_plot_url:
            dist_plot_url += cache_key
        if scatter_plot_url:
            scatter_plot_url += cache_key

    except Exception as e:
        return render(request, 'project1/partials/explore_partial.html', {
            'error': f"Failed to load dataset: {str(e)}"
        })

    # Retrieve and clear temporary errors if set by post views
    error = request.session.pop('clean_error', None)

    context = {
        'dataset_id': dataset_id,
        'summary_data': summary_data,
        'columns': columns,
        'selected_dist_col': selected_dist_col,
        'dist_plot_url': dist_plot_url,
        'x_col': x_col,
        'y_col': y_col,
        'hue_col': hue_col,
        'scatter_plot_url': scatter_plot_url,
        'error': error,
    }

    return render(request, 'project1/partials/explore_partial.html', context)


@require_POST
def clean_reset_view(request):
    dataset_id = request.session.get('active_dataset_id')
    if not dataset_id:
        return explore_view(request)

    dataset = Dataset.objects.get(id=dataset_id)
    _, error = reset_dataset(dataset)
    
    if error:
        request.session['clean_error'] = error
        
    return explore_view(request)


@require_POST
def clean_drop_nulls_view(request):
    dataset_id = request.session.get('active_dataset_id')
    if not dataset_id:
        return explore_view(request)

    dataset = Dataset.objects.get(id=dataset_id)
    _, error = drop_missing_rows(dataset)

    if error:
        request.session['clean_error'] = error

    return explore_view(request)


@require_POST
def clean_fill_missing_view(request):
    dataset_id = request.session.get('active_dataset_id')
    if not dataset_id:
        return explore_view(request)

    dataset = Dataset.objects.get(id=dataset_id)
    _, error = impute_missing_values(dataset)

    if error:
        request.session['clean_error'] = error

    return explore_view(request)


@require_POST
def clean_filter_distribution_view(request):
    dataset_id = request.session.get('active_dataset_id')
    if not dataset_id:
        return explore_view(request)

    dataset = Dataset.objects.get(id=dataset_id)
    column = request.POST.get('dist_col')
    min_value = request.POST.get('min_value')
    max_value = request.POST.get('max_value')

    _, error = filter_out_of_distribution(dataset, column, min_value, max_value)

    if error:
        request.session['clean_error'] = error

    # Re-use explore_view so all plot URLs and summaries rebuild automatically
    return explore_view(request)


def train_view(request):
    """ View called when clicking the Training tab """
    dataset_id = request.session.get('active_dataset_id')
    return HttpResponse(f"<p>Training configured for Dataset ID: {dataset_id}</p>")
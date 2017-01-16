$(document).ready(function() {
    // Show the default selected field
    selected_feat_id = $('#feature_selector').find(":selected").val()
    feat_id = "#feature-" + selected_feat_id;
    graph_id = "#graph-" + selected_feat_id;
    $(feat_id).removeClass('table-hidden');
    $(graph_id).removeClass('table-hidden');

    // Respond to changing the selected feature (univariate)
    $('#feature_selector').on('change', function() {
        // Hide all tables
        $('.feature-tables').addClass('table-hidden');
        $('.feature-graphs').addClass('table-hidden');

        // Show the tables for the selected field
        feat_id = "#feature-" + $(this).val();
        graph_id = "#graph-" + $(this).val();
        $(feat_id).removeClass('table-hidden');
        $(graph_id).removeClass('table-hidden');
    });

    // Show the default selected interaction
    selected_int_id = $('#interaction_selector').find(":selected").val()
    interaction_id = "#interactions-" + selected_int_id;
    interactiongraph_id = "#interactiongraphs-" + selected_int_id;
    $(interaction_id).removeClass('table-hidden');
    $(interactiongraph_id).removeClass('table-hidden');

    // Respond to changing the selected feature (bivariate)
    $('#interaction_selector').on('change', function() {
        // Hide all tables
        $('.interactions-tables').addClass('table-hidden');
        $('.interactions-graphs').addClass('table-hidden');

        // Show the tables for the selected field
        feat_id = "#interactions-" + $(this).val();
        graph_id = "#interactiongraphs-" + $(this).val();
        $(feat_id).removeClass('table-hidden');
        $(graph_id).removeClass('table-hidden');
    });

    // Show active tab
    var url_parts = location.href.split('/');
    var last_segment = url_parts[url_parts.length-1];
    $('.navbar-nav a[href="' + last_segment + '"]').parents('li').addClass('active');

    $.validator.addMethod('filesize', function (value, element, param) {
        return this.optional(element) || (element.files[0].size <= param)
    }, 'File size must be less than {0}');

    // Validate the upload form
    $("#fileUpload").validate({
        // Specify validation rules
        rules: {
          title: "required",
          file: {
            required: true,
            extension: "csv|tsv|json|xls|xlsx",
            filesize: 16777216,
          }
        },
        // Specify validation error messages
        messages: {
          title: "Please enter the title of the data set",
          file: {
            required: "Please select the data set file",
            extension: "Accepted file types are csv, tsv, json, xlsx, xls",
            filesize: "Files must be under 2MB"
          },
        },
        submitHandler: function(form) {
            form.submit();
        }
    });
});